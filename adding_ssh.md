# WSL SSH Remote Access Setup
*Access your Windows/WSL desktop from a laptop over LAN*

---

## Overview

| Component | Value |
|---|---|
| Desktop LAN IP | `192.168.86.250` |
| WSL internal IP | `172.29.255.231` (changes on restart — see Step 4) |
| Forwarded port | `2222` (Windows → WSL port 22) |
| SSH user | run `whoami` in WSL to confirm |

---

## Step 1 — Install & Start OpenSSH in WSL

```bash
sudo apt install openssh-server
sudo service ssh start
sudo service ssh enable   # auto-start
sudo service ssh status   # verify
```

---

## Step 2 — Get WSL Internal IP

```bash
ip addr show eth0 | grep 'inet '
# inet 172.29.255.231/20 ...
```

Note this IP — it changes on every WSL restart (see Step 4).

---

## Step 3 — Port Forward Windows → WSL (Admin PowerShell)

```powershell
# Forward port 2222 → WSL port 22
netsh interface portproxy add v4tov4 `
    listenport=2222 listenaddress=0.0.0.0 `
    connectport=22 connectaddress=172.29.255.231

# Open firewall
New-NetFirewallRule -DisplayName "WSL SSH" `
    -Direction Inbound -Protocol TCP -LocalPort 2222 -Action Allow

# Verify
netsh interface portproxy show all
```

> The firewall rule only needs to be added once. The portproxy rule must be re-run when WSL IP changes (Step 4).

---

## Step 4 — Fix Dynamic WSL IP (Startup Script)

Save as `C:\Users\<YourUser>\wsl-ssh-forward.ps1`:

```powershell
$wslIP = (wsl hostname -I).Trim().Split()[0]
netsh interface portproxy delete v4tov4 listenport=2222 listenaddress=0.0.0.0
netsh interface portproxy add v4tov4 `
    listenport=2222 listenaddress=0.0.0.0 `
    connectport=22 connectaddress=$wslIP
Write-Host "Forwarding to $wslIP"
```

Schedule via **Task Scheduler**:
- Trigger: At log on
- Program: `powershell.exe`
- Arguments: `-ExecutionPolicy Bypass -WindowStyle Hidden -File "C:\Users\<YourUser>\wsl-ssh-forward.ps1"`
- Run with highest privileges: ✓

---

## Step 5 — Connect from Laptop

```bash
ssh -p 2222 <wsl_username>@192.168.86.250
```

Optional `~/.ssh/config` on laptop:
```
Host desktop-wsl
    HostName 192.168.86.250
    Port 2222
    User <wsl_username>
```
Then just: `ssh desktop-wsl`

---

## What happens after a desktop restart

Three things break on reboot and need to recover in order:

1. **WSL SSH service stops.** WSL doesn't auto-start services on Windows boot. The Task Scheduler script restarts the portproxy but not SSH itself. Fix: add a second Task Scheduler entry that runs:
   ```
   wsl sudo service ssh start
   ```
   Or add `[boot] command = service ssh start` to `/etc/wsl.conf` in WSL (requires WSL2 with systemd disabled).

2. **WSL IP changes.** Already handled by the Task Scheduler script in Step 4.

3. **Any running processes are gone.** `python serve.py`, anything running in a terminal — all dead. You'll need to SSH back in and restart them manually. Consider using `tmux` or `screen` to keep sessions persistent across reconnects (though not across reboots).

---

## Connecting from the laptop

Once setup is complete, from your laptop terminal:

```bash
ssh desktop-wsl        # if you added ~/.ssh/config (recommended)
# or
ssh -p 2222 <wsl_username>@192.168.86.250
```

**To avoid typing your password every time**, run this once from the laptop:

```bash
ssh-copy-id -p 2222 <wsl_username>@192.168.86.250
```

This copies your laptop's public key to the desktop. After that, login is passwordless.

**For a full editor experience (recommended):** install the **Remote - SSH** extension in VS Code on your laptop. Connect to `desktop-wsl` and VS Code runs its server on the desktop — you get the full editor, file tree, terminal, and debugger, but all execution happens on the desktop's hardware and filesystem.

---

## Desktop LAN IP stability

The desktop LAN IP (`192.168.86.250`) is assigned by your router via DHCP and can change after a router restart. If `ssh desktop-wsl` suddenly stops working with "connection timed out", this is likely why.

**Fix:** log into your router admin UI and reserve that IP address for your desktop's MAC address (sometimes called "DHCP reservation" or "static lease"). The desktop keeps the same IP permanently without any changes to Windows or WSL.

---

## Persistence

SSH gives direct in-place access to the WSL filesystem. Changes made over SSH are immediately visible locally on the desktop and vice versa — no syncing.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Connection refused | SSH not running: `sudo service ssh start` |
| Connection timed out | Stale portproxy rule — re-run Step 4 script |
| Permission denied | Wrong username — confirm with `whoami` in WSL |
| Works today, broken tomorrow | WSL IP changed — check Task Scheduler ran the script |

---

## Optional — Tailscale (Cross-Network Access)

The above works on LAN only. For access from anywhere with no router port forwarding: install [Tailscale](https://tailscale.com) on both machines, log in with the same account, SSH to the Tailscale IP. Done.