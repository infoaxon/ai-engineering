# agent_no_llm.py

import sqlite3
import argparse


def send_nudge(to: str, body: str):
    """Stub delivery function-replace with email/SMS integration."""
    print(f"\n--- NUDGE ---\nTo: {to}\n{body}\n")


def main(db_path: str, segment: str):
    # 1) Connect
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 2) Fetch the template for this segment
    cur.execute(
        """
        SELECT templateText
        FROM NudgeTemplates
        WHERE segmentName = ?
    """,
        (segment,),
    )
    tmpl_row = cur.fetchone()
    if not tmpl_row:
        conn.close()
        raise ValueError(f"No template found for segment '{segment}'")
    template = tmpl_row["templateText"]

    # 3) Fetch customers + dynamic attributes via join
    cur.execute(
        """
        SELECT
            CM.recordId,
            CM.name,
            CM.email,
            CM.age,
            CM.numOfLogins30d,
            CM.totalCustomerLifetimeValue,
            CM.propensityScore,
            CMD.daysSinceLastPolicy
        FROM CustomerMaster CM
        JOIN CustomerMasterWithDays CMD USING(recordId)
        JOIN CustomerSegments CS USING(recordId)
        WHERE CS.segmentName = ?
    """,
        (segment,),
    )
    customers = cur.fetchall()

    # 4) For each customer, format & send
    for cust in customers:
        body = template.format(
            name=cust["name"],
            age=cust["age"],
            numOfLogins30d=cust["numOfLogins30d"],
            totalCustomerLifetimeValue=cust["totalCustomerLifetimeValue"],
            propensityScore=cust["propensityScore"],
            daysSinceLastPolicy=cust["daysSinceLastPolicy"],
        )
        send_nudge(cust["email"], body)

    conn.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Send pre-defined nudges for a given customer segment"
    )
    p.add_argument(
        "--db", default="cdp_segments_demo.db", help="Path to SQLite database file"
    )
    p.add_argument(
        "--segment", required=True, help="Segment name to process (e.g. AtRiskLapsed)"
    )
    args = p.parse_args()

    main(args.db, args.segment)
