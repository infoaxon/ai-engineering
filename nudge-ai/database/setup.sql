PRAGMA foreign_keys = ON;

-- 1) CustomerMaster: flattened view of all customer attributes
CREATE TABLE IF NOT EXISTS CustomerMaster (
  recordId                       TEXT      PRIMARY KEY,
  name                           TEXT      NOT NULL,
  email                          TEXT      NOT NULL,
  age                            INTEGER,
  numOfLogins30d                 INTEGER,
  totalCustomerLifetimeValue     REAL,
  propensityScore                REAL,
  lastPolicyEndDate              TEXT      -- ISO date string
);

-- 2) CustomerSegments: one row per customer with assigned segment
CREATE TABLE IF NOT EXISTS CustomerSegments (
  recordId       TEXT NOT NULL,
  segmentName    TEXT NOT NULL,
  daysSinceLastPolicy  INTEGER,
  FOREIGN KEY (recordId) REFERENCES CustomerMaster(recordId)
);

-- 3) NudgeTemplates: one approved template per segment
CREATE TABLE IF NOT EXISTS NudgeTemplates (
  segmentName    TEXT      PRIMARY KEY,
  templateText   TEXT      NOT NULL
);

-- 4) Populate NudgeTemplates with your brand-approved copy
INSERT OR REPLACE INTO NudgeTemplates(segmentName, templateText) VALUES
  ('AtRiskLapsed',    'Hi {name}, it has now been {daysSinceLastPolicy} days since your last policy-let us help you get back on track!'),
  ('HighValueEngaged','Hey {name}, thanks for logging in {numOfLogins30d} times-check out our VIP offers tailored for you!'),
  ('YoungUrban',      'Hi {name}, protect your urban lifestyle today-policies start at just ₹499/month!'),
  ('FamilyBundlers',  'Hello {name}, bundle your familys life and health coverage for extra savings-{numOfLogins30d} people are already enrolled!'),
  ('DormantProspect', 'Hey {name}, we noticed you started a quote-lock in your rate now before it expires!'),
  ('SeniorStrained',  'Hi {name}, ask about our flexible payment plan-no more worries after {daysSinceLastPolicy} days of lapse!'),
  ('CampaignResponsive','{name}, thanks for engaging {numOfLogins30d} times-unlock your special discount today!')
;

-- 5) Example: Create a VIEW to compute daysSinceLastPolicy on the fly
CREATE VIEW IF NOT EXISTS CustomerMasterWithDays AS
SELECT
  CM.*,
  CAST(julianday('now') - julianday(CM.lastPolicyEndDate) AS INTEGER) AS daysSinceLastPolicy
FROM CustomerMaster CM;

-- 6) Example: Materialize a segment assignment based on some criteria
-- (In practice these INSERTs come from your ETL or Python scripts)
INSERT OR REPLACE INTO CustomerSegments(recordId, segmentName, daysSinceLastPolicy)
SELECT
  recordId,
  'AtRiskLapsed',
  daysSinceLastPolicy
FROM CustomerMasterWithDays
WHERE daysSinceLastPolicy > 30
  AND propensityScore < 0.3;

INSERT OR REPLACE INTO CustomerSegments(recordId, segmentName, daysSinceLastPolicy)
SELECT
  recordId,
  'HighValueEngaged',
  daysSinceLastPolicy
FROM CustomerMasterWithDays
WHERE totalCustomerLifetimeValue >= 100000
  AND numOfLogins30d >= 10;

-- 7) (Optional) Index segments for fast lookups
CREATE INDEX IF NOT EXISTS idx_segments_name ON CustomerSegments(segmentName);

