.headers on
.mode column
SELECT CS.segmentName, COUNT(*) AS cnt
FROM CustomerSegments CS
GROUP BY CS.segmentName;

