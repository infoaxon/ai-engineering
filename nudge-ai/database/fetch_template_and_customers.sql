-- Template
SELECT templateText FROM NudgeTemplates WHERE segmentName='AtRiskLapsed';

-- Customers
SELECT CM.name, CM.email, CS.daysSinceLastPolicy
FROM CustomerMasterWithDays CM
JOIN CustomerSegments CS USING(recordId)
WHERE CS.segmentName = 'AtRiskLapsed'
LIMIT 5;
