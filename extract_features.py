# extract_features.py

import psycopg2
import pandas as pd

def extract_aact_data(conn_info, output_csv="data/aact_extracted.csv"):
    query = """
    SELECT
        s.nct_id,
        sp.name AS sponsor_name,
        sp.agency_class AS sponsor_type,
        s.study_type,
        s.phase,
        s.enrollment,
        s.enrollment_type,
        s.start_date,
        s.completion_date,
        s.overall_status,
        s.why_stopped,
        d.allocation,
        d.masking,
        d.primary_purpose,
        s.number_of_arms,
        e.gender,
        e.minimum_age,
        e.maximum_age,
        e.criteria AS eligibility_criteria,
        array_agg(DISTINCT c.name) AS conditions,
        array_agg(DISTINCT i.name) AS interventions,
        array_agg(DISTINCT i.intervention_type) AS intervention_types
    FROM
        ctgov.studies s
    LEFT JOIN ctgov.sponsors sp
        ON s.nct_id = sp.nct_id AND sp.lead_or_collaborator = 'lead'
    LEFT JOIN ctgov.designs d
        ON s.nct_id = d.nct_id
    LEFT JOIN ctgov.eligibilities e
        ON s.nct_id = e.nct_id
    LEFT JOIN ctgov.conditions c
        ON s.nct_id = c.nct_id
    LEFT JOIN ctgov.interventions i
        ON s.nct_id = i.nct_id
    WHERE
        s.start_date IS NOT NULL
    GROUP BY
        s.nct_id, sp.name, sp.agency_class, s.study_type, s.phase,
        s.enrollment, s.enrollment_type, s.start_date, s.completion_date,
        s.overall_status, s.why_stopped,
        d.allocation, d.masking, d.primary_purpose,
        s.number_of_arms,
        e.gender, e.minimum_age, e.maximum_age, e.criteria
    ORDER BY
        sponsor_name, start_date;
    """
    try:
        print("Connecting to AACT database...")
        conn = psycopg2.connect(**conn_info)
        df = pd.read_sql_query(query, conn)
        df.to_csv(output_csv, index=False)
        print(f"✅ Extracted {len(df)} records to {output_csv}")
    except Exception as e:
        print("❌ Failed to extract data:", e)
    finally:
        if 'conn' in locals():
            conn.close()
