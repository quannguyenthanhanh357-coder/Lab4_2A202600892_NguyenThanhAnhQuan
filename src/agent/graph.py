from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""Bạn là trợ lý đặt hàng thiết bị điện tử chuyên nghiệp. Hôm nay là {current_day}.

=== QUY TRÌNH BẮT BUỘC ===
Khi khách hàng yêu cầu tạo đơn hàng hợp lệ, bạn PHẢI gọi đúng 5 công cụ theo thứ tự sau:
  1. list_products       — tìm sản phẩm trong catalog
  2. get_product_details — lấy giá, tồn kho và detail_token
  3. get_discount        — lấy mã giảm giá (dùng email khách hàng làm seed_hint)
  4. calculate_order_totals — tính tổng tiền (cần detail_token và discount_rate từ bước trên)
  5. save_order          — lưu đơn hàng cuối cùng

=== ĐIỀU KIỆN TIÊN QUYẾT (kiểm tra TRƯỚC khi gọi bất kỳ công cụ nào) ===
Yêu cầu đặt hàng hợp lệ phải có ĐẦY ĐỦ 5 thông tin sau:
  - Tên khách hàng (customer_name)
  - Số điện thoại (customer_phone)
  - Email (customer_email)
  - Địa chỉ giao hàng (shipping_address)
  - Ít nhất 1 sản phẩm kèm số lượng

Nếu THIẾU BẤT KỲ thông tin nào trong 5 mục trên → hỏi lại ngay, KHÔNG gọi công cụ nào.

=== GUARDRAIL — TỪ CHỐI NGAY (không gọi công cụ) ===
Từ chối mọi yêu cầu thuộc một trong các loại sau:
  - Tạo hóa đơn giả hoặc đơn hàng không có thật
  - Yêu cầu bỏ qua hoặc sửa thủ công mức giảm giá
  - Yêu cầu bỏ qua kiểm tra tồn kho
  - Bất kỳ hành động nào vi phạm catalog hoặc chính sách cửa hàng

=== QUY TẮC VỀ DỮ LIỆU ===
  - Chỉ dùng product_id, giá, tồn kho, token, tổng tiền từ KẾT QUẢ công cụ — không bịa đặt
  - Chỉ lưu đơn hàng SAU KHI calculate_order_totals trả về status "ok"
  - Nếu tồn kho không đủ → báo lỗi, KHÔNG lưu đơn

=== TRẢ LỜI ===
  - Luôn trả lời bằng tiếng Việt, ngắn gọn, súc tích
  - Sau khi lưu thành công: xác nhận mã đơn, mã giảm giá, tổng tiền sau giảm, đường dẫn file
  - Khi thiếu thông tin: chỉ hỏi đúng phần còn thiếu
  - Khi từ chối: giải thích ngắn gọn lý do từ chối
""".strip()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def build_tools(store: OrderDataStore):
    """
    Five strongly-typed tools using Pydantic schemas.
    The detail_token flows from get_product_details → calculate_order_totals → save_order
    to prevent the model from skipping steps or fabricating data.
    """

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """
        Search the electronics product catalog.
        Use product names, brands, or keywords as `query`.
        Filter by category (laptop, monitor, mouse, keyboard, headphone, dock, storage, stand, webcam).
        Returns product_id, name, brand, category, tags.
        IMPORTANT: Call get_product_details next with the chosen product_ids to get price, stock, and detail_token.
        """
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """
        Return exact price, stock, warranty, and SKU for the given product_ids.
        Also returns a `detail_token` — a validation token that MUST be passed unchanged to
        calculate_order_totals and save_order. Never modify or fabricate this token.
        """
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """
        Retrieve the campaign discount for this order.
        Use the customer's email as seed_hint (fallback to phone number).
        Returns discount_rate (0.1 or 0.2) and campaign_code.
        Pass discount_rate and campaign_code unchanged to calculate_order_totals and save_order.
        """
        return json.dumps(
            store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier),
            ensure_ascii=False,
        )

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
    ) -> str:
        """
        Validate stock levels and compute the discounted order total.
        Requires:
          - items: list of {product_id, quantity} using exact IDs from get_product_details
          - detail_token: token returned by get_product_details (do not alter)
          - discount_rate: value returned by get_discount (must be 0.1 or 0.2)
        Returns subtotal, discount_amount, and final_total.
        If status is "error", do NOT call save_order — report the issue to the customer instead.
        """
        payload = store.calculate_order_totals(
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """
        Persist the final order to a local JSON file.
        Only call this AFTER calculate_order_totals returns status "ok".
        All arguments must come from previous tool outputs — do not fabricate any values.
        Returns order_id, save path, and the complete saved_order payload.
        """
        result = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(
        data_dir or DEFAULT_DATA_DIR,
        output_dir or DEFAULT_OUTPUT_DIR,
        today=today,
    )
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    tools = build_tools(store)
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=build_system_prompt(today or store.today),
    )


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response

    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)

    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_final_answer(messages) -> str:
    """Return the last non-empty AI message content."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Convert tool call / tool result pairs into a grading-friendly trace."""
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tc in getattr(message, "tool_calls", []) or []:
                pending[tc["id"]] = {
                    "name": tc["name"],
                    "args": tc.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    # Flush any pending calls that have no result yet
    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))

    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Parse the save_order tool output into (saved_order, path)."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
