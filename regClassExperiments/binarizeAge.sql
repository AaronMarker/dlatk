
-- Use the variable in the main query
CREATE VIEW survey_outcomes_waves_agg_v9_more4_Black_Dep_BinAge AS
SELECT 
    *,
    CASE 
        WHEN age > 36 THEN 1
        ELSE 0
    END AS age_binarized
FROM 
    survey_outcomes_waves_aggregated_v9_floats_more4_Black_Dep;