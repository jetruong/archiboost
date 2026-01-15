"""
Integration tests for the API endpoints.
"""

import io
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Adjust Python path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_pdf_bytes():
    """
    Create a minimal valid PDF for testing.
    This is a simple PDF with one blank page.
    """
    # Minimal PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000206 00000 n 
trailer
<< /Size 5 /Root 1 0 R >>
startxref
300
%%EOF
"""
    return pdf_content


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_info(self, client):
        """Test that root returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestUploadEndpoint:
    """Tests for upload endpoint."""
    
    def test_upload_missing_files(self, client):
        """Test upload fails without files."""
        response = client.post("/api/v1/upload")
        
        assert response.status_code == 422  # Validation error
    
    def test_upload_invalid_content_type(self, client):
        """Test upload fails with non-PDF files."""
        files = {
            "file_a": ("test.txt", b"not a pdf", "text/plain"),
            "file_b": ("test2.txt", b"also not a pdf", "text/plain"),
        }
        
        response = client.post("/api/v1/upload", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_FILE_TYPE"


class TestSessionEndpoint:
    """Tests for session endpoint."""
    
    def test_get_nonexistent_session(self, client):
        """Test getting a session that doesn't exist."""
        response = client.get("/api/v1/sessions/nonexistent-session-id")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["code"] == "SESSION_NOT_FOUND"


class TestImageEndpoint:
    """Tests for image serving endpoint."""
    
    def test_get_nonexistent_image(self, client):
        """Test getting an image that doesn't exist."""
        response = client.get("/api/v1/images/nonexistent-image-id")
        
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["code"] == "IMAGE_NOT_FOUND"


class TestPreviewEndpoint:
    """Tests for preview endpoint."""
    
    def test_preview_nonexistent_session(self, client):
        """Test preview for nonexistent session."""
        response = client.get("/api/v1/sessions/nonexistent/preview?which=A")
        
        assert response.status_code == 404
    
    def test_preview_invalid_which(self, client):
        """Test preview with invalid 'which' parameter."""
        response = client.get("/api/v1/sessions/test-session/preview?which=C")
        
        assert response.status_code == 422  # Validation error
    
    def test_preview_invalid_dpi(self, client):
        """Test preview with out-of-range DPI."""
        response = client.get("/api/v1/sessions/test-session/preview?which=A&dpi=10")
        
        assert response.status_code == 422  # Validation error
