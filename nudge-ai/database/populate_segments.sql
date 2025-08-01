-- HighValueEngaged
INSERT OR REPLACE INTO CustomerSegments(recordId, segmentName, daysSinceLastPolicy)
SELECT
  recordId,
  'HighValueEngaged' AS segmentName,
  daysSinceLastPolicy
FROM CustomerMasterWithDays
WHERE totalCustomerLifetimeValue >= 100000
  AND numOfLogins30d >= 10;

-- AtRiskLapsed
INSERT OR REPLACE INTO CustomerSegments(recordId, segmentName, daysSinceLastPolicy)
SELECT
  recordId,
  'AtRiskLapsed' AS segmentName,
  daysSinceLastPolicy
FROM CustomerMasterWithDays
WHERE daysSinceLastPolicy > 30
  AND propensityScore < 0.3;
