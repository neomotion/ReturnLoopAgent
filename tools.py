import sqlite3
from pathlib import Path

DATA = Path(__file__).parent / "data"
db = sqlite3.connect(DATA / "returnloop.db")
db.row_factory = sqlite3.Row

def one(sql, *args):
    row = db.execute(sql, args).fetchone()
    return dict(row) if row else None

# ---- tools ----
def get_claim(claim_id):
    # the claim + its order (never the expected label)
    row = one("""select k.id, k.type, k.text, o.id as order_id, o.customer_id, o.product, o.price
                 from claims k join orders o on o.id = k.order_id where k.id = ?""", claim_id)
    return row or {"error": f"no claim {claim_id}"}

def get_customer(customer_id):
    row = one("select * from customers where id = ?", customer_id)
    if not row:
        return {"error": f"no customer {customer_id}"}
    row["return_rate"] = round(row["past_returns"] / row["past_orders"], 2) if row["past_orders"] else 0
    return row

def get_delivery(order_id):
    return one("select * from deliveries where order_id = ?", order_id) or {"error": f"no delivery {order_id}"}

def get_policy():
    return {"policy": (DATA / "policy.md").read_text(encoding="utf-8")}

FUNCTIONS = {"get_claim": get_claim, "get_customer": get_customer,
             "get_delivery": get_delivery, "get_policy": get_policy}

def tool(name, description, **params):
    # build one tool schema; every param is a required string
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object",
                       "properties": {p: {"type": "string", "description": d} for p, d in params.items()},
                       "required": list(params)}}}

TOOLS = [
    tool("get_claim", "Get a return claim and its order (product, price, customer_id).", claim_id="claim id, e.g. k1"),
    tool("get_customer", "Get a customer's account age, past orders, past returns and return rate.",
         customer_id="customer id, e.g. c1"),
    tool("get_delivery", "Get carrier status for an order: delivered / lost / in_transit, proof, days since delivery.",
         order_id="order id, e.g. o1"),
    tool("get_policy", "Get the full return and refund policy."),
]
