package org.hc.cbl.entity;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.Date;

import static org.junit.jupiter.api.Assertions.*;

class ApplicantTest {
    private Applicant applicant;
    private Date testDate;

    @BeforeEach
    void setUp() {
        applicant = new Applicant();
        testDate = new Date();
    }

    @Test
    void testBasicProperties() {
        // Arrange
        String name = "John Doe";
        String reference = "APP123";
        String gender = "Male";
        String maritalStatus = "Single";
        String ethnicity = "White British";
        String primaryLanguage = "English";

        // Act
        applicant.setName(name);
        applicant.setReference(reference);
        applicant.setDateOfBirth(testDate);
        applicant.setGender(gender);
        applicant.setMaritalStatus(maritalStatus);
        applicant.setEthnicity(ethnicity);
        applicant.setPrimaryLanguage(primaryLanguage);

        // Assert
        assertEquals(name, applicant.getName());
        assertEquals(reference, applicant.getReference());
        assertEquals(testDate, applicant.getDateOfBirth());
        assertEquals(gender, applicant.getGender());
        assertEquals(maritalStatus, applicant.getMaritalStatus());
        assertEquals(ethnicity, applicant.getEthnicity());
        assertEquals(primaryLanguage, applicant.getPrimaryLanguage());
    }

    @Test
    void testEmploymentDetailsRelationship() {
        // Arrange
        ArrayList<ApplicantEmploymentDetail> employmentDetails = new ArrayList<>();
        ApplicantEmploymentDetail detail = new ApplicantEmploymentDetail();
        employmentDetails.add(detail);

        // Act
        applicant.setEmploymentDetails(employmentDetails);

        // Assert
        assertNotNull(applicant.getEmploymentDetails());
        assertEquals(1, applicant.getEmploymentDetails().size());
        assertSame(detail, applicant.getEmploymentDetails().get(0));
    }

    @Test
    void testBidsRelationship() {
        // Arrange
        ArrayList<Bid> bids = new ArrayList<>();
        Bid bid = new Bid();
        bids.add(bid);

        // Act
        applicant.setBids(bids);

        // Assert
        assertNotNull(applicant.getBids());
        assertEquals(1, applicant.getBids().size());
        assertSame(bid, applicant.getBids().get(0));
    }

    @Test
    void testCurrentPropertyRelationship() {
        // Arrange
        Property property = new Property();
        property.setReference("PROP123");

        // Act
        applicant.setCurrentProperty(property);

        // Assert
        assertNotNull(applicant.getCurrentProperty());
        assertEquals("PROP123", applicant.getCurrentProperty().getReference());
    }
} 