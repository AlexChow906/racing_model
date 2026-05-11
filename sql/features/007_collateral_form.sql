CREATE OR REPLACE TABLE f007 AS
WITH
-- Step 1: For every horse_history entry, pre-compute what the horse did in its next 3 races
-- Uses LEAD window function (no correlated subquery, fast)
subsequent AS (
    SELECT
        history_id, horse_id, race_id, scheduled_off_utc,
        finishing_position, won, field_size,
        LEAD(CASE WHEN won THEN 1 ELSE 0 END, 1) OVER w AS next1_won,
        LEAD(CASE WHEN won THEN 1 ELSE 0 END, 2) OVER w AS next2_won,
        LEAD(CASE WHEN won THEN 1 ELSE 0 END, 3) OVER w AS next3_won,
        LEAD(CASE WHEN finishing_position <= 3 THEN 1 ELSE 0 END, 1) OVER w AS next1_placed,
        LEAD(CASE WHEN finishing_position <= 3 THEN 1 ELSE 0 END, 2) OVER w AS next2_placed,
        LEAD(CASE WHEN finishing_position <= 3 THEN 1 ELSE 0 END, 3) OVER w AS next3_placed,
        LEAD(scheduled_off_utc, 1) OVER w AS next1_off,
        LEAD(scheduled_off_utc, 2) OVER w AS next2_off,
        LEAD(scheduled_off_utc, 3) OVER w AS next3_off
    FROM horse_history
    WHERE finishing_position IS NOT NULL
    WINDOW w AS (PARTITION BY horse_id ORDER BY scheduled_off_utc)
),

-- Step 2: Current runners with their decision cutoff
base AS (
    SELECT ru.runner_id, ru.race_id, ru.horse_id, ra.decision_cutoff_utc
    FROM runners ru
    JOIN races ra ON ra.race_id = ru.race_id
),

-- Step 3: Horse's last 5 races
past_races AS (
    SELECT
        b.runner_id, b.horse_id, b.decision_cutoff_utc,
        hh.race_id AS past_race_id,
        hh.scheduled_off_utc AS past_off,
        hh.finishing_position AS our_pos,
        ROW_NUMBER() OVER (PARTITION BY b.runner_id ORDER BY hh.scheduled_off_utc DESC) AS rn
    FROM base b
    JOIN horse_history hh
        ON hh.horse_id = b.horse_id
       AND hh.scheduled_off_utc < b.decision_cutoff_utc
       AND hh.finishing_position IS NOT NULL
),

-- Step 4: Opponents in those past races that our horse BEAT
-- Join to subsequent to get their future performance
beaten_opp AS (
    SELECT
        pr.runner_id,
        pr.decision_cutoff_utc,
        pr.past_race_id,
        s.horse_id AS opp_horse_id,
        -- Only count subsequent results that happened before our decision cutoff (no leakage)
        CASE WHEN s.next1_off IS NOT NULL AND s.next1_off < pr.decision_cutoff_utc THEN s.next1_won END AS n1w,
        CASE WHEN s.next2_off IS NOT NULL AND s.next2_off < pr.decision_cutoff_utc THEN s.next2_won END AS n2w,
        CASE WHEN s.next3_off IS NOT NULL AND s.next3_off < pr.decision_cutoff_utc THEN s.next3_won END AS n3w,
        CASE WHEN s.next1_off IS NOT NULL AND s.next1_off < pr.decision_cutoff_utc THEN s.next1_placed END AS n1p,
        CASE WHEN s.next2_off IS NOT NULL AND s.next2_off < pr.decision_cutoff_utc THEN s.next2_placed END AS n2p,
        CASE WHEN s.next3_off IS NOT NULL AND s.next3_off < pr.decision_cutoff_utc THEN s.next3_placed END AS n3p
    FROM past_races pr
    JOIN subsequent s
        ON s.race_id = pr.past_race_id
       AND s.horse_id != pr.horse_id
       AND s.finishing_position > pr.our_pos
    WHERE pr.rn <= 5
),

-- Step 5: Per-opponent subsequent win/place rate
opp_scores AS (
    SELECT
        runner_id,
        opp_horse_id,
        past_race_id,
        (COALESCE(n1w,0) + COALESCE(n2w,0) + COALESCE(n3w,0))::DOUBLE
            / GREATEST(1, (CASE WHEN n1w IS NOT NULL THEN 1 ELSE 0 END
                         + CASE WHEN n2w IS NOT NULL THEN 1 ELSE 0 END
                         + CASE WHEN n3w IS NOT NULL THEN 1 ELSE 0 END))
            AS opp_sub_win_rate,
        (COALESCE(n1p,0) + COALESCE(n2p,0) + COALESCE(n3p,0))::DOUBLE
            / GREATEST(1, (CASE WHEN n1p IS NOT NULL THEN 1 ELSE 0 END
                         + CASE WHEN n2p IS NOT NULL THEN 1 ELSE 0 END
                         + CASE WHEN n3p IS NOT NULL THEN 1 ELSE 0 END))
            AS opp_sub_place_rate,
        CASE WHEN (n1w = 1 OR n2w = 1 OR n3w = 1) THEN 1 ELSE 0 END AS opp_went_on_to_win
    FROM beaten_opp
    WHERE (n1w IS NOT NULL OR n2w IS NOT NULL OR n3w IS NOT NULL)
),

-- Step 6: Aggregate per runner
agg AS (
    SELECT
        runner_id,
        AVG(opp_sub_win_rate) AS collateral_beaten_win_rate,
        AVG(opp_sub_place_rate) AS collateral_beaten_place_rate,
        SUM(opp_went_on_to_win) AS collateral_franked_winners,
        COUNT(DISTINCT opp_horse_id) AS collateral_beaten_count
    FROM opp_scores
    GROUP BY 1
)

SELECT
    b.runner_id,
    b.race_id,
    a.collateral_beaten_win_rate,
    a.collateral_beaten_place_rate,
    COALESCE(a.collateral_franked_winners, 0) AS collateral_franked_winners,
    COALESCE(a.collateral_beaten_count, 0) AS collateral_beaten_count,
    b.decision_cutoff_utc - INTERVAL 1 SECOND AS event_timestamp_utc,
    b.decision_cutoff_utc
FROM base b
LEFT JOIN agg a ON a.runner_id = b.runner_id;
