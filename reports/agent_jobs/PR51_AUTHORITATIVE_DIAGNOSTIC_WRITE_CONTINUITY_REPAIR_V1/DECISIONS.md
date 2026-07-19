- Keep packet-root and every output-domain descriptor owned through final
  descriptor-relative readback, manifest/signature verification, and close.
- Stage fully in-domain, register rollback ownership immediately after create
  and publication, verify exact surfaces plus identity/size/SHA, and roll back
  across all domains on every exception path.
- Bind all four domains before diagnostic writes and reject device/inode aliases.
- Reproducibility descriptor and physical output hashes remain unchanged.
