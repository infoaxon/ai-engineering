// config.js
export const config = {
  Qs: [
    [
      {
        label: "Details fetched from {{documentType}}",
        key: "KYCDETAILS",
        qGroup: 1,
        mandatory: false,
        editable: true,
        display: true,
        type: "kyc-details",
        placeholder: "Enter details...",
        source: [
          { key: "0", displayText: "Male" },
          { key: "1", displayText: "Female" },
          { key: "2", displayText: "Male Transgender" },
          { key: "3", displayText: "Female Transgender" }
        ]
      },
      {
        label: "Email Address",
        key: "EMAIL",
        qGroup: 1,
        mandatory: true,
        display: true,
        type: "email",
        placeholder: "Enter your email",
        valueValidationLabel: "Please enter a valid email address"
      },
      {
        label: "Occupation",
        key: "OCCUPATION",
        qGroup: 1,
        mandatory: true,
        display: true,
        type: "select",
        source: [
          { key: "doctor", displayText: "Doctor" },
          { key: "engineer", displayText: "Engineer" },
          { key: "teacher", displayText: "Teacher" }
        ]
      }
    ]
  ]
};
