# agent_no_llm_improved.py

import sqlite3
import argparse


def send_nudge(to: str, body: str):
    """Stub delivery function-swap in your email/SMS logic here."""
    print(f"\n--- NUDGE ---\nTo: {to}\n{body}\n")


def main(db_path: str, segment: str):
    # 1) Open connection
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 2) Fetch the template from NudgeTemplates
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
    print(f"[DEBUG] Using template for '{segment}':\n{template}\n")

    # 3) Query customer attributes, joining in all relevant tables
    cur.execute(
        """
        SELECT
            CM.recordId,
            CM.name,
            CM.email,
            CM.age,
            UP.numOfLogins30d,
            LM.totalCustomerLifetimeValue,
            LM.propensityScore,
            CMD.daysSinceLastPolicy
        FROM CustomerMaster CM
        -- computed days view
        JOIN CustomerMasterWithDays CMD USING(recordId)
        -- segment assignment
        JOIN CustomerSegments CS USING(recordId)
        -- behavioral data
        LEFT JOIN UserActionProfile UP USING(recordId)
        -- value & propensity
        LEFT JOIN LifetimeMetrics LM USING(recordId)
        WHERE CS.segmentName = ?
    """,
        (segment,),
    )
    customers = [dict(r) for r in cur.fetchall()]
    conn.close()

    # 4) For each customer, format & send the nudge
    for cust in customers:
        # Ensure daysSinceLastPolicy is present
        cust["daysSinceLastPolicy"] = int(cust["daysSinceLastPolicy"])
        # Fill in placeholders; missing keys will raise KeyError
        try:
            message = template.format_map(cust)
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(
                f"Template refers to '{{{missing}}}' but available keys are: {list(cust.keys())}"
            )
        send_nudge(cust["email"], message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send stored nudges for a given customer segment"
    )
    parser.add_argument(
        "--db", default="cdp_segments_demo.db", help="SQLite database file path"
    )
    parser.add_argument(
        "--segment", required=True, help="Segment name to process (e.g. AtRiskLapsed)"
    )
    args = parser.parse_args()
    main(args.db, args.segment)
