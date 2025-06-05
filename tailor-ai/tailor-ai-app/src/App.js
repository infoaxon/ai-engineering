// App.js
import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';

// Inline configuration JSON
const config = {
  Qs: [
    [
      {
        label: 'Details fetched from {{documentType}}',
        key: 'KYCDETAILS',
        display: true,
        type: 'kyc-details',
        placeholder: 'Enter details...',
      },
      {
        label: 'Email Address',
        key: 'EMAIL',
        display: true,
        type: 'email',
        placeholder: 'Enter your email',
        mandatory: true,
      },
      {
        label: 'Occupation',
        key: 'OCCUPATION',
        display: true,
        type: 'select',
        mandatory: true,
        source: [
          { key: 'doctor', displayText: 'Doctor' },
          { key: 'engineer', displayText: 'Engineer' },
          { key: 'teacher', displayText: 'Teacher' },
        ],
      },
    ],
  ],
};

// Single field renderer
const FormField = ({ field, value, onChange }) => {
  const commonProps = {
    id: field.key,
    name: field.key,
    placeholder: field.placeholder || '',
    value: value || '',
    onChange: (e) => onChange(field.key, e.target.value),
    style: { width: '100%', padding: 8, marginBottom: 16 },
  };

  switch (field.type) {
    case 'kyc-details':
      return (
        <div key={field.key}>
          <label htmlFor={field.key} style={{ display: 'block', marginBottom: 4 }}>
            {field.label}
          </label>
          <textarea {...commonProps} rows={4} />
        </div>
      );

    case 'email':
      return (
        <div key={field.key}>
          <label htmlFor={field.key} style={{ display: 'block', marginBottom: 4 }}>
            {field.label}{field.mandatory && ' *'}
          </label>
          <input type="email" required={field.mandatory} {...commonProps} />
        </div>
      );

    case 'select':
      return (
        <div key={field.key}>
          <label htmlFor={field.key} style={{ display: 'block', marginBottom: 4 }}>
            {field.label}{field.mandatory && ' *'}
          </label>
          <select
            {...commonProps}
            onChange={(e) => onChange(field.key, e.target.value)}
          >
            <option value="">Select</option>
            {field.source.map((opt) => (
              <option key={opt.key} value={opt.key}>
                {opt.displayText}
              </option>
            ))}
          </select>
        </div>
      );

    default:
      return null;
  }
};

// Main dynamic form component
const DynamicForm = () => {
  const [formData, setFormData] = useState({});
  const [submitted, setSubmitted] = useState(false);

  const handleChange = (key, val) => {
    setFormData((prev) => ({ ...prev, [key]: val }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmitted(true);
    console.log('Form Data:', formData);
  };

  // Flatten Qs to a single array
  const fields = (config.Qs || []).flat();

  return (
    <div style={{ maxWidth: 600, margin: '2rem auto' }}>
      <form onSubmit={handleSubmit} style={{ display: submitted ? 'none' : 'block' }}>
        {fields.map((f) => f.display && (
          <FormField
            key={f.key}
            field={f}
            value={formData[f.key]}
            onChange={handleChange}
          />
        ))}
        <button type="submit" style={{ padding: '0.75rem 1.5rem' }}>Submit</button>
      </form>

      {submitted && (
        <div style={{ marginTop: 20 }}>
          <h3>Submitted Data</h3>
          <pre>{JSON.stringify(formData, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

// Bootstrap React
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<DynamicForm />);

