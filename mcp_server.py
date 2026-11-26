import sqlite3
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from photos import PHOTOS, dhash, distance
from search import search

# tools run in worker threads; read-only, so sharing one connection is safe
db = sqlite3.connect(Path(__file__).parent / "data" / "returnloop.db", check_same_thread=False)
db.row_factory = sqlite3.Row
mcp = MCPServer("returnloop")

def one(sql, *args):
    row = db.execute(sql, args).fetchone()
    return dict(row) if row else None

# ---- tools: type hints + docstring become the schema the model sees ----
@mcp.tool()
def get_claim(claim_id: str) -> dict:
    """Get a return claim (type, customer text) and its order (product, price, customer_id). claim_id like k1."""
    row = one("""select k.id, k.type, k.text, k.photo, o.id as order_id, o.customer_id, o.product, o.category, o.price
                 from claims k join orders o on o.id = k.order_id where k.id = ?""", claim_id)
    return row or {"error": f"no claim {claim_id}"}

@mcp.tool()
def get_customer(customer_id: str) -> dict:
    """Get a customer's account age in days, past orders, past returns and return rate. customer_id like c1."""
    row = one("select * from customers where id = ?", customer_id)
    if not row:
        return {"error": f"no customer {customer_id}"}
    row["return_rate"] = round(row["past_returns"] / row["past_orders"], 2) if row["past_orders"] else 0
    return row

@mcp.tool()
def get_delivery(order_id: str) -> dict:
    """Get carrier status for an order: delivered / lost / in_transit, proof, days since delivery. order_id like o1."""
    return one("select * from deliveries where order_id = ?", order_id) or {"error": f"no delivery {order_id}"}

@mcp.tool()
def check_photo_reuse(claim_id: str) -> dict:
    """Check if the claim's photo was already used in an earlier claim (perceptual hash, 0 = same picture)."""
    row = one("select photo from claims where id = ?", claim_id)
    if not row or not row["photo"]:
        return {"has_photo": False}
    h = dhash(PHOTOS / row["photo"])
    past = {r["claim_id"]: distance(h, r["hash"]) for r in db.execute("select * from past_photos")}
    return {"has_photo": True, "reused_from": [c for c, d in past.items() if d <= 8], "distances": past}

@mcp.tool()
def search_policy(query: str) -> dict:
    """Search the merchant's return policy. Returns the 3 best sections, each with an id to cite."""
    return {"results": search(query)}

if __name__ == "__main__":
    mcp.run()   # stdio: the agent starts this file as a subprocess
