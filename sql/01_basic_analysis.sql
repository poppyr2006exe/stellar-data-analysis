SELECT
    class,
    COUNT(*) AS number_of_objects,
    ROUND(
        COUNT(*) * 100.0 /
        (SELECT COUNT(*) FROM observations),
        2
    ) AS percentage
FROM observations
GROUP BY class
ORDER BY number_of_objects DESC;

-- Average redshift by class

SELECT
    class,
    ROUND(AVG(redshift), 4) AS mean_redshift
FROM observations
GROUP BY class
ORDER BY mean_redshift DESC;


-- Objects with redshift greater than 1

SELECT
    class,
    COUNT(*) AS number_of_objects
FROM observations
WHERE redshift > 1
GROUP BY class
ORDER BY number_of_objects DESC;


-- Average colour indices by class

SELECT
    class,
    ROUND(AVG(u_g), 3) AS mean_u_g,
    ROUND(AVG(g_r), 3) AS mean_g_r,
    ROUND(AVG(r_i), 3) AS mean_r_i,
    ROUND(AVG(i_z), 3) AS mean_i_z
FROM observations
GROUP BY class
ORDER BY class;

-- Percentage of each class with redshift greater than 1

SELECT
    class,
    COUNT(*) AS total_objects,
    SUM(CASE WHEN redshift > 1 THEN 1 ELSE 0 END) AS high_redshift_objects,
    ROUND(
        SUM(CASE WHEN redshift > 1 THEN 1 ELSE 0 END) * 100.0
        / COUNT(*),
        2
    ) AS percentage_high_redshift
FROM observations
GROUP BY class
ORDER BY percentage_high_redshift DESC;