package org.hc.cbl.entity;

import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.junit.jupiter.SpringExtension;

@ExtendWith(SpringExtension.class)
@DataJpaTest
@ActiveProfiles("test")
public abstract class BaseEntityTest {
    // Common test utilities and helper methods can be added here
    
    protected void clearDatabase() {
        // Add implementation if needed
    }
    
    protected <T> void validateEntity(T entity) {
        // Add implementation if needed
    }
} 