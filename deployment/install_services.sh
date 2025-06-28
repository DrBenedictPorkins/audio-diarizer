#!/bin/bash
# Install systemd services for production deployment

set -e

echo "🔧 Installing Audio Diarization API systemd services..."

# Check if running as root/sudo
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run with sudo"
    exit 1
fi

# Copy service files
cp audio-diarizer-api.service /etc/systemd/system/
cp audio-diarizer-worker.service /etc/systemd/system/

# Set permissions
chmod 644 /etc/systemd/system/audio-diarizer-api.service
chmod 644 /etc/systemd/system/audio-diarizer-worker.service

# Reload systemd
systemctl daemon-reload

# Enable services (don't start yet)
systemctl enable audio-diarizer-api.service
systemctl enable audio-diarizer-worker.service

echo "✅ Services installed and enabled"
echo ""
echo "To start the services:"
echo "sudo systemctl start audio-diarizer-api"
echo "sudo systemctl start audio-diarizer-worker"
echo ""
echo "To check status:"
echo "sudo systemctl status audio-diarizer-api"
echo "sudo systemctl status audio-diarizer-worker"
echo ""
echo "To view logs:"
echo "sudo journalctl -u audio-diarizer-api -f"
echo "sudo journalctl -u audio-diarizer-worker -f"