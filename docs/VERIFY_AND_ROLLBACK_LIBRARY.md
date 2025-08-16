# Verify & Rollback Library

## Overview
Reusable verify commands and a 7-step rollback template for maintaining system integrity.

## Verify Commands

### Endpoints
Verify specific API endpoint operations:
```bash
curl -X GET http://localhost:5000/api/status
```

### Calibration Metrics
Validate model calibration through metrics like Brier score and reliability graphs.

### Leakage Checks
Ensure no features leak information from the target outcome post-prediction.

### Database Versions
Check the current database version matches expected schema:
```sql
SELECT version FROM alembic_version;
```

### Logs Verification
Ensure key process logs are being created and logged without errors.

### Archive Confirmations
Verify all required files are properly archived as per retention policies.

## 7-Step Rollback Template

1. **Initiation**
   - Identify the need for rollback.
   - Inform stakeholders and freeze new changes.

2. **Scope Definition**
   - Determine the affected components or modules.
   - Assess extent and impact of rollback.

3. **Backup**
   - Create backups of current configurations or databases.
   - Ensure backups are accessible and verified.

4. **Preparation**
   - Gather all necessary scripts and files for rollback.
   - Alert all affected teams & departments.

5. **Execution**
   - Perform rollback systematically using verified scripts.
   - Monitor logs for any errors or warnings.

6. **Verification**
   - Verify rollback success through tests and stakeholder review.
   - Check system functionality & data integrity.

7. **Closure**
   - Document the rollback incident.
   - Conduct a root cause analysis and update policies.
   - Restart normal operations and inform all teams.
