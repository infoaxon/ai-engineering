package org.hc.cbl.entity;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class PartnerTest {
    private Partner partner;

    @BeforeEach
    void setUp() {
        partner = new Partner();
    }

    @Test
    void testBasicProperties() {
        // Arrange
        String name = "Test Partner";
        String code = "PART123";
        String contactPerson = "John Smith";
        String email = "john@partner.com";
        String phone = "1234567890";

        // Act
        partner.setName(name);
        partner.setCode(code);
        partner.setContactPerson(contactPerson);
        partner.setEmail(email);
        partner.setPhone(phone);

        // Assert
        assertEquals(name, partner.getName());
        assertEquals(code, partner.getCode());
        assertEquals(contactPerson, partner.getContactPerson());
        assertEquals(email, partner.getEmail());
        assertEquals(phone, partner.getPhone());
    }

    @Test
    void testPropertiesRelationship() {
        // Arrange
        List<Property> properties = new ArrayList<>();
        Property property = new Property();
        property.setReference("PROP123");
        properties.add(property);

        // Act
        partner.setProperties(properties);

        // Assert
        assertNotNull(partner.getProperties());
        assertEquals(1, partner.getProperties().size());
        assertEquals("PROP123", partner.getProperties().get(0).getReference());
    }

    @Test
    void testBidsRelationship() {
        // Arrange
        List<Bid> bids = new ArrayList<>();
        Bid bid = new Bid();
        bids.add(bid);

        // Act
        partner.setBids(bids);

        // Assert
        assertNotNull(partner.getBids());
        assertEquals(1, partner.getBids().size());
        assertSame(bid, partner.getBids().get(0));
    }
} 