package org.hc.cbl.entity;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class PropertyTest {
    private Property property;

    @BeforeEach
    void setUp() {
        property = new Property();
    }

    @Test
    void testBasicProperties() {
        // Arrange
        String reference = "PROP123";
        String address = "123 Main St";
        String postcode = "AB12 3CD";
        Integer bedrooms = 2;
        String propertyType = "Apartment";
        Double rentAmount = 1200.00;
        String councilTaxBand = "B";
        String energyEfficiencyRating = "C";

        // Act
        property.setReference(reference);
        property.setAddress(address);
        property.setPostcode(postcode);
        property.setBedrooms(bedrooms);
        property.setPropertyType(propertyType);
        property.setRentAmount(rentAmount);
        property.setCouncilTaxBand(councilTaxBand);
        property.setEnergyEfficiencyRating(energyEfficiencyRating);

        // Assert
        assertEquals(reference, property.getReference());
        assertEquals(address, property.getAddress());
        assertEquals(postcode, property.getPostcode());
        assertEquals(bedrooms, property.getBedrooms());
        assertEquals(propertyType, property.getPropertyType());
        assertEquals(rentAmount, property.getRentAmount());
        assertEquals(councilTaxBand, property.getCouncilTaxBand());
        assertEquals(energyEfficiencyRating, property.getEnergyEfficiencyRating());
    }

    @Test
    void testBidsRelationship() {
        // Arrange
        List<Bid> bids = new ArrayList<>();
        Bid bid = new Bid();
        bids.add(bid);

        // Act
        property.setBids(bids);

        // Assert
        assertNotNull(property.getBids());
        assertEquals(1, property.getBids().size());
        assertSame(bid, property.getBids().get(0));
    }

    @Test
    void testLandlordRelationship() {
        // Arrange
        Landlord landlord = new Landlord();
        landlord.setName("Test Landlord");

        // Act
        property.setLandlord(landlord);

        // Assert
        assertNotNull(property.getLandlord());
        assertEquals("Test Landlord", property.getLandlord().getName());
    }

    @Test
    void testPropertyAttributesRelationship() {
        // Arrange
        List<PropertyAttribute> attributes = new ArrayList<>();
        PropertyAttribute attribute = new PropertyAttribute();
        attribute.setAttributeName("Parking");
        attribute.setAttributeValue("Yes");
        attributes.add(attribute);

        // Act
        property.setAttributes(attributes);

        // Assert
        assertNotNull(property.getAttributes());
        assertEquals(1, property.getAttributes().size());
        assertEquals("Parking", property.getAttributes().get(0).getAttributeName());
        assertEquals("Yes", property.getAttributes().get(0).getAttributeValue());
    }
} 