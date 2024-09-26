CREATE VIEW survey_outcomes_waves_aggregated_v9_floats_more4_Black_Dep AS
SELECT o.*, mode_table.depression_past_any
FROM survey_outcomes_waves_aggregated_v9_floats_moreThan4_withBlack o
LEFT JOIN (
    SELECT f.user_id, f.depression_past_any
    FROM (
        SELECT user_id, depression_past_any, COUNT(*) AS freq
        FROM bl_wave_v9
        GROUP BY user_id, depression_past_any
    ) AS f
    INNER JOIN (
        SELECT user_id, MAX(freq) AS max_freq
        FROM (
            SELECT user_id, depression_past_any, COUNT(*) AS freq
            FROM bl_wave_v9
            GROUP BY user_id, depression_past_any
        ) AS freq_table
        GROUP BY user_id
    ) AS max_freq_table
    ON f.user_id = max_freq_table.user_id AND f.freq = max_freq_table.max_freq
) AS mode_table
ON o.user_id = mode_table.user_id
