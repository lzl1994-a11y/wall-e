# WALL-E systemd drop-in

The walle.service.d/40-microphone-auto.conf file captures the existing runtime
override for /root/wall-e-bt with --remote-control. It marks the default TROS
environment as ready only after its setup script succeeds.

After syncing this repository on a host that already has walle.service,
install the drop-in and reload systemd:

    sudo install -D -m 0644 deploy/systemd/walle.service.d/40-microphone-auto.conf /etc/systemd/system/walle.service.d/40-microphone-auto.conf
    sudo systemctl daemon-reload

This does not enable or start the service. Review the path and launch
arguments before using the template on a different host.
