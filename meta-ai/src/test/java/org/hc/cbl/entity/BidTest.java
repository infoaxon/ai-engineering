package org.hc.cbl.entity;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Date;

import static org.junit.jupiter.api.Assertions.*;

class BidTest {
    private Bid bid;
    private Date testDate;

    @BeforeEach
    void setUp() {
        bid = new Bid();
        testDate = new Date();
    }

    @Test
    void testBasicProperties() {
        // Arrange
        Date bidDate = testDate;
        Integer bidPosition = 1;
        String bidStatus = "Active";
        String bidSource = "Online";
        Boolean shortlisted = true;

        // Act
        bid.setBidDate(bidDate);
        bid.setBidPosition(bidPosition);
        bid.setBidStatus(bidStatus);
        bid.setBidSource(bidSource);
        bid.setShortlisted(shortlisted);

        // Assert
        assertEquals(bidDate, bid.getBidDate());
        assertEquals(bidPosition, bid.getBidPosition());
        assertEquals(bidStatus, bid.getBidStatus());
        assertEquals(bidSource, bid.getBidSource());
        assertEquals(shortlisted, bid.getShortlisted());
    }

    @Test
    void testApplicantRelationship() {
        // Arrange
        Applicant applicant = new Applicant();
        applicant.setName("John Doe");

        // Act
        bid.setApplicant(applicant);

        // Assert
        assertNotNull(bid.getApplicant());
        assertEquals("John Doe", bid.getApplicant().getName());
    }

    @Test
    void testPropertyRelationship() {
        // Arrange
        Property property = new Property();
        property.setReference("PROP123");

        // Act
        bid.setProperty(property);

        // Assert
        assertNotNull(bid.getProperty());
        assertEquals("PROP123", bid.getProperty().getReference());
    }

    @Test
    void testPartnerRelationship() {
        // Arrange
        Partner partner = new Partner();
        partner.setName("Test Partner");

        // Act
        bid.setPartner(partner);

        // Assert
        assertNotNull(bid.getPartner());
        assertEquals("Test Partner", bid.getPartner().getName());
    }
} 