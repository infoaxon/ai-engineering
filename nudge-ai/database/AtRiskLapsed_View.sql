CREATE VIEW AtRiskLapsed AS
SELECT
  CM.recordId,
  CM.name,
  CM.email,
  CM.age,
  CM.numOfLogins30d,
  CM.totalCustomerLifetimeValue,
  CM.propensityScore,
  CM.lastPolicyEndDate,
  CSeg.daysSinceLastPolicy
FROM CustomerMasterWithDays CM
JOIN CustomerSegments CSeg
  ON CM.recordId = CSeg.recordId
WHERE CSeg.segmentName = 'AtRiskLapsed';
