CREATE TABLE CustomerMaster (
  recordId                   TEXT    PRIMARY KEY,
  name                       TEXT    NOT NULL,
  email                      TEXT    NOT NULL,
  age                        INTEGER,
  numOfLogins30d             INTEGER,
  totalCustomerLifetimeValue REAL,
  propensityScore            REAL,
  segmentFlag                TEXT,       -- new column
  lastPolicyEndDate          TEXT        -- ISO date string
);
