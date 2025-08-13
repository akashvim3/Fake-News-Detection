#!/bin/bash

# SSL certificate setup script

echo "🔒 Setting up SSL certificates for FactCheck AI..."

# Create SSL directory
mkdir -p ssl

# Check if certificates already exist
if [ -f "ssl/cert.pem" ] && [ -f "ssl/key.pem" ]; then
    echo "✅ SSL certificates already exist"
    exit 0
fi

# Generate self-signed certificate for development
echo "🔧 Generating self-signed SSL certificate for development..."

openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
    -subj "/C=US/ST=State/L=City/O=FactCheck AI/CN=localhost"

if [ $? -eq 0 ]; then
    echo "✅ Self-signed SSL certificate generated successfully"
    echo ""
    echo "📁 Certificate files:"
    echo "   - ssl/cert.pem"
    echo "   - ssl/key.pem"
    echo ""
    echo "⚠️  For production, replace with certificates from a trusted CA"
    echo "   (Let's Encrypt, DigiCert, etc.)"
else
    echo "❌ Failed to generate SSL certificate"
    exit 1
fi
