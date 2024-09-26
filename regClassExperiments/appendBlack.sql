CREATE VIEW survey_outcomes_waves_aggregated_v9_floats_moreThan4_withBlack AS
SELECT o.*, mode_table.is_black
FROM survey_outcomes_waves_aggregated_v9_floats_moreThan4 o
LEFT JOIN (
    SELECT f.user_id, f.is_black
    FROM (
        SELECT user_id, is_black, COUNT(*) AS freq
        FROM bl_wave_v9
        GROUP BY user_id, is_black
    ) AS f
    INNER JOIN (
        SELECT user_id, MAX(freq) AS max_freq
        FROM (
            SELECT user_id, is_black, COUNT(*) AS freq
            FROM bl_wave_v9
            GROUP BY user_id, is_black
        ) AS freq_table
        GROUP BY user_id
    ) AS max_freq_table
    ON f.user_id = max_freq_table.user_id AND f.freq = max_freq_table.max_freq
) AS mode_table
ON o.user_id = mode_table.user_id
