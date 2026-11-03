"""book-rag: a local, hybrid-search RAG system over your personal book library."""

__version__ = "1.0.0"

# Use the operating system's certificate store for HTTPS (model downloads from Hugging Face).
# This fixes "SSL: CERTIFICATE_VERIFY_FAILED" behind antivirus / corporate TLS inspection.
try:  # pragma: no cover - optional dependency
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass
