"""
tools/faq_tool.py
LangChain tool for answering common e-commerce FAQs.
This is a simple keyword-matching tool — no LLM needed for FAQs,
which saves tokens and makes responses faster.
"""
from langchain.tools import tool

# Pre-written answers to the most common customer questions
FAQ_DATABASE = {
    "return": (
        "Return Policy: You can return any item within 30 days of delivery. "
        "The item must be unused and in its original packaging. "
        "To start a return, go to My Orders > Select Order > Request Return."
    ),
    "refund": (
        "Refunds are processed within 5-7 business days after we receive the "
        "returned item. The amount is credited to your original payment method. "
        "For UPI payments, refunds take 2-3 business days."
    ),
    "shipping": (
        "Shipping Info: Free shipping on all orders above Rs 999. "
        "Standard delivery: 3-5 business days. "
        "Express delivery (1-2 days): Rs 199 extra. "
        "Same-day delivery available in select cities for Rs 299."
    ),
    "cancel": (
        "Cancellation: Orders can be cancelled within 24 hours of placing them. "
        "After dispatch, you cannot cancel — but you can request a return once delivered. "
        "To cancel: My Orders > Select Order > Cancel Order."
    ),
    "payment": (
        "Payment Methods: We accept UPI (GPay, PhonePe, Paytm), "
        "Credit/Debit cards (Visa, Mastercard, RuPay), Net Banking, "
        "and Cash on Delivery (COD) for orders below Rs 5000. "
        "All payments are secured with 256-bit SSL encryption."
    ),
    "warranty": (
        "Warranty: Electronics come with a 1-year manufacturer warranty. "
        "Clothing and accessories have a 6-month warranty against manufacturing defects. "
        "To claim warranty, contact us with your order ID and product photos."
    ),
    "contact": (
        "Customer Support: Email us at support@shopai.in "
        "or call 1800-123-4567 (toll-free, available 9 AM - 9 PM, Mon-Sat). "
        "Average response time is 2-4 hours on email."
    ),
    "track": (
        "Order Tracking: You can track your order in My Orders section. "
        "A tracking link is also sent to your registered email and phone "
        "once the order is shipped."
    ),
    "exchange": (
        "Exchange Policy: We offer free exchanges within 30 days for size or "
        "color issues. To exchange, go to My Orders > Request Exchange. "
        "The replacement will be shipped within 2-3 days of receiving your return."
    ),
}


@tool
def faq_lookup(question: str) -> str:
    """
    Answer frequently asked questions about returns, refunds, shipping,
    cancellations, payments, warranty, tracking, or exchanges.
    Use this for general questions that do NOT need an order ID.
    Input should be the customer's question as a string.
    """
    question_lower = question.lower()

    # Aliases — map common variant words to FAQ keys
    ALIASES = {
        "upi": "payment", "gpay": "payment", "paytm": "payment",
        "phonepay": "payment", "credit card": "payment", "debit card": "payment",
        "cod": "payment", "cash on delivery": "payment",
        "deliver": "shipping", "dispatch": "shipping",
        "broken": "return", "defective": "return", "damaged": "return",
        "money back": "refund",
        "cancell": "cancel", "cancel": "cancel",
        "guarantee": "warranty", "repair": "warranty",
        "track my": "track",
    }
    for alias, key in ALIASES.items():
        if alias in question_lower:
            return FAQ_DATABASE.get(key, "")

    # Find the best matching FAQ topic
    for keyword, answer in FAQ_DATABASE.items():
        if keyword in question_lower:
            return answer

    # If no match found, give a helpful fallback
    return (
        "I don't have a specific answer for that. "
        "Please contact our support team:\n"
        "  Email : support@shopai.in\n"
        "  Phone : 1800-123-4567 (toll-free)\n"
        "  Hours : 9 AM - 9 PM, Monday to Saturday"
    )
