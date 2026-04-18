# WSL SSH Setup — Bidirectional
*SSH between two Windows/WSL machines on the same LAN*

---

## Understanding SSH

### What SSH actually is

SSH (Secure Shell) is a protocol for logging into another machine's terminal over a network, as if you were sitting at it. Once connected, you're in a shell on the remote machine — you can run commands, edit files, start servers. Everything executes there, not on your local machine.

```
your terminal  ──SSH──►  remote machine's shell
(local input)             (remote execution)
```

### Passwords vs. keys

SSH supports two ways to prove your identity:

**Password auth** — you type a password each time. Simple but tedious and less secure.

**Key auth** — you generate a key *pair*: a private key and a public key. You put the public key on the remote machine, keep the private key on your machine. When you connect, SSH uses cryptography to prove you own the private key without ever sending it. The remote machine checks whether your public key is in its trust list. If yes, you're in — no password prompt.

The analogy: the public key is a padlock you give to the remote machine. The private key is the only key that opens it. You can hand out padlocks freely; only you can open them.

### Key pair anatomy

When you run `ssh-keygen`, it creates two files:

```
~/.ssh/id_ed25519        ← private key  (never share this, never copy it around)
~/.ssh/id_ed25519.pub    ← public key   (safe to share freely)
```

`ed25519` is the algorithm — it's the modern standard. You may also see `id_rsa` on older systems.

---

## The .ssh folder explained

Every SSH-capable user has a `~/.ssh/` directory. Here's what each file does:

### `id_ed25519` — your private key
Your identity for **outbound** connections. When you `ssh` into another machine, this is what proves who you are. Keep it on this machine only. Permissions must be `600` (owner read/write only) — SSH will refuse to use it if it's world-readable.

### `id_ed25519.pub` — your public key
The shareable half of your key pair. When you want to log into a remote machine, you copy this file's contents into that machine's `authorized_keys`. It's just a single line of text starting with `ssh-ed25519 AAAA...`.

### `authorized_keys` — your trust list
Controls who can log **into** this machine as you. Each line is one public key. When someone SSHes in, their client offers their private key; SSH checks whether the corresponding public key is in this file. If it is, they're authenticated.

```
Think of it as: "I trust anyone who owns the private key matching these public keys."
```

Permissions must be `600`.

### `config` — outbound connection shortcuts
Settings for connections you make **from** this machine. Without it, you'd type `ssh -p 2222 cosenzac@192.168.86.250` every time. With it, you define a named host and just type `ssh desktop`.

A typical entry:
```
Host desktop
  HostName 192.168.86.250   ← IP or domain to connect to
  Port 2222                 ← non-default port
  User cosenzac             ← username on the remote machine
  IdentityFile ~/.ssh/id_ed25519  ← which private key to use
```

The `config` file has no effect on who can connect *in* to your machine — that's `authorized_keys`'s job.

### `known_hosts` — remote machine fingerprints
The first time you SSH to a new machine, you see:
```
The authenticity of host '192.168.86.250' can't be established.
Are you sure you want to continue connecting? (yes/no)?
```
Saying yes records the remote machine's fingerprint here. On future connections SSH silently checks that the fingerprint still matches — if it doesn't (e.g. the machine was replaced or someone is intercepting the connection), it refuses to connect and warns you. This is protection against man-in-the-middle attacks.

### Permissions summary

```
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 600 ~/.ssh/authorized_keys
chmod 600 ~/.ssh/config
```

SSH is paranoid about permissions and will silently fail or refuse to use keys if they're too permissive.

---

## Directionality

This is the most confusing part of SSH. Every file has a direction:

| File | Lives on | Controls |
|---|---|---|
| `id_ed25519` (private) | **source** machine | Proves source's identity when connecting out |
| `id_ed25519.pub` (public) | **destination** machine (in `authorized_keys`) | Tells destination to trust the source |
| `config` | **source** machine | Shortcuts for connections made from here |
| `authorized_keys` | **destination** machine | Who is allowed to log in here |
| `known_hosts` | **source** machine | Fingerprints of destinations this machine has visited |

To set up A → B:
1. Generate a key pair on **A**
2. Copy A's **public key** to B's `authorized_keys`
3. Add B to A's `config`

To also set up B → A (bidirectional):
1. Generate a key pair on **B** (or reuse existing)
2. Copy B's **public key** to A's `authorized_keys`
3. Add A to B's `config`

Each direction is independent. Having A→B working tells you nothing about whether B→A works.

---

## Machine reference

| | This machine (laptop) | Desktop |
|---|---|---|
| Windows LAN IP | `192.168.86.21` | `192.168.86.250` |
| WSL internal IP | `172.28.215.211` (changes — see below) | `172.29.255.231` (changes — see below) |
| WSL username | `zaccosenza` | `cosenzac` |
| SSH port | `2222` | `2222` |

Both machines use **systemd-enabled WSL2** (`/etc/wsl.conf` has `[boot] systemd=true`).

### The WSL connection chain

WSL runs inside a Hyper-V virtual network and isn't directly reachable from the outside. Windows acts as a gateway:

```
WSL (source) → Windows (source) → Windows (destination) → WSL (destination)
```

The `portproxy` rule on the destination's Windows side is what makes this work — it listens on the Windows IP and forwards traffic to the WSL internal IP.

---

## Step 1 — Install and configure SSH server (run on each machine's WSL)

```bash
sudo apt-get install -y openssh-server

# Use port 2222 to avoid conflicts with Windows's own SSH on port 22
sudo tee /etc/ssh/sshd_config.d/wsl.conf << 'EOF'
Port 2222
PubkeyAuthentication yes
PasswordAuthentication no
AuthorizedKeysFile .ssh/authorized_keys
EOF

sudo systemctl enable ssh
sudo systemctl start ssh

# Verify it's listening
ss -tlnp | grep 2222
```

---

## Step 2 — Exchange public keys

Each machine needs the other's public key in its `authorized_keys`.

**Print your public key:**
```bash
cat ~/.ssh/id_ed25519.pub
```

**Add the other machine's key:**
```bash
echo "ssh-ed25519 AAAA...their-key..." >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

Gotcha: **the username matters**. The key goes into the `authorized_keys` of whichever user you're logging in *as* on the destination. If the destination username differs from the source (e.g. `zaccosenza` connecting to a machine where you're `cosenzac`), auth fails with `Permission denied` even if the key is correct. Always verify with `whoami` on the destination.

---

## Step 3 — Windows port forwarding (run on each machine as Administrator)

Run in **cmd as Administrator**:

```cmd
netsh interface portproxy delete v4tov4 listenport=2222 listenaddress=0.0.0.0
netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=2222 connectaddress=<WSL-internal-IP>
netsh advfirewall firewall delete rule name="WSL SSH"
netsh advfirewall firewall add rule name="WSL SSH" dir=in action=allow protocol=TCP localport=2222
```

Replace `<WSL-internal-IP>` with the output of `hostname -I | awk '{print $1}'` in WSL.

The firewall rule only needs to be added once. The portproxy rule must be re-run whenever the WSL IP changes (see Step 4).

---

## Step 4 — WSL IP stability

### Why the IP changes

WSL2 uses a Hyper-V virtual network adapter (`vEthernet (WSL (Hyper-V firewall))`). Windows assigns a new subnet to this adapter on each boot — there's no guarantee it gets the same address. The portproxy rule hardcodes the WSL IP, so it goes stale after every reboot.

Check the current WSL IP from Windows:
```cmd
wsl hostname -I
```

Or from within WSL:
```bash
hostname -I | awk '{print $1}'
```

### Fix: auto-refresh on Windows login

Save this as `C:\Users\<YourUser>\wsl-ssh-forward.ps1`:

```powershell
$wslIp = (wsl hostname -I).Trim().Split(" ")[0]
netsh interface portproxy delete v4tov4 listenport=2222 listenaddress=0.0.0.0
netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=2222 connectaddress=$wslIp
```

Register it as a scheduled task (run in **PowerShell as Administrator**):

```powershell
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-WindowStyle Hidden -ExecutionPolicy Bypass -File C:\Users\<YourUser>\wsl-ssh-forward.ps1"
$trigger = New-ScheduledTaskTrigger -AtLogOn
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -RunLevel Highest
Register-ScheduledTask -TaskName "WSL SSH Forward" -Action $action -Trigger $trigger -Principal $principal -Force
```

This runs at every Windows login, reads the fresh WSL IP, and updates the portproxy rule.

### What still breaks on reboot

1. **WSL IP changes** — handled by the Task Scheduler script above.
2. **SSH service stops** — with systemd enabled, SSH auto-starts when WSL boots. WSL boots the first time any WSL terminal opens after Windows login. To ensure SSH is available before you open a terminal, add a second Task Scheduler entry running `wsl.exe -e sudo systemctl start ssh`.
3. **Running processes are gone** — servers, scripts, anything in a terminal is killed on reboot. Use `tmux` to keep sessions alive across reconnects (though not across full reboots).

### Alternative: fix the WSL IP permanently

With `networkingMode=mirrored` in `C:\Users\<YourUser>\.wslconfig`, WSL shares the Windows network interface — no separate WSL IP, no portproxy needed. Requires Windows 11 or recent Win10 insider builds.

```ini
[wsl2]
networkingMode=mirrored
```

---

## Step 5 — SSH config shortcuts

On **this machine** (`~/.ssh/config`):

```
Host desktop
  HostName 192.168.86.250
  Port 2222
  User cosenzac
  IdentityFile ~/.ssh/id_ed25519
```

Connect with:
```bash
ssh desktop
```

---

## VSCode Remote-SSH

### What it does

The **Remote - SSH** extension lets VSCode run its backend on a remote machine over SSH. When connected, the VSCode window on your local machine is just a UI shell — the file tree, terminal, language servers, debugger, and extensions all run on the remote machine. You're editing and running code there, not locally.

This is the best way to use VSCode with SSH because:
- The terminal inside VSCode is a shell on the remote machine
- Files open directly from the remote filesystem — no mounting, no syncing
- Extensions that need to run code (Python, ESLint, etc.) run on the remote, where the code actually lives
- Your local VSCode UI settings and themes carry over; language extensions are installed on the remote side

### Setup

1. Install the **Remote - SSH** extension (`ms-vscode-remote.remote-ssh`) in VSCode on your local machine
2. Make sure your `~/.ssh/config` has the host defined (Step 5 above)
3. Open the command palette: `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → select `desktop`
4. VSCode opens a new window and installs its server on the remote machine (one-time, takes ~30 seconds)
5. You're now in a full VSCode session running on the desktop

### What the connected session looks like

- The bottom-left corner shows `SSH: desktop` in green — that's your indicator you're in a remote session
- **File → Open Folder** opens folders on the **remote** machine's filesystem
- The integrated terminal (`Ctrl+`\`) opens a shell on the **remote** machine
- Extensions you install in the remote session are installed on the remote (Python extension needs to be there, not just locally)
- Your `.venv`, project files, Claude Code — everything on the remote is accessible as if you were sitting at it

### Reconnecting

VSCode remembers recent remote connections. To reconnect:
- Click the green `><` icon in the bottom-left corner → **Connect to Host** → `desktop`
- Or `Ctrl+Shift+P` → **Remote-SSH: Connect to Host**

### Port forwarding inside VSCode

If you're running a server on the remote (e.g. `python -m http.server 8080`) and want to access it in your local browser, VSCode can forward the port automatically. In the **Ports** tab (next to Terminal), click **Forward a Port** and enter `8080`. VSCode tunnels it to your local machine — open `localhost:8080` in your browser and it hits the remote server.

---

## Desktop LAN IP stability

The Windows LAN IP (`192.168.86.x`) is assigned by your router via DHCP and can change after a router restart. If SSH suddenly stops with "connection timed out", this is likely why.

**Fix:** log into your router admin UI and create a DHCP reservation for each machine's MAC address (sometimes called "static lease"). The machine keeps the same LAN IP permanently with no changes to Windows or WSL.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` | SSH not running on destination | `sudo systemctl start ssh` in destination WSL |
| `Connection timed out` | Stale portproxy rule or wrong IP | Re-run Step 3 with current WSL IP |
| `Permission denied (publickey)` | Wrong username or key not in `authorized_keys` | Verify username with `whoami`; check `~/.ssh/authorized_keys` on destination |
| Works today, broken tomorrow | WSL IP changed, Task Scheduler didn't run | Run `wsl-ssh-forward.ps1` manually |
| VSCode says "could not establish connection" | Same as above, or SSH server not running | Check SSH status, check portproxy |
| VSCode connects but terminal is wrong machine | You opened a local window, not remote | Look for `SSH: desktop` in bottom-left corner |

---

## Optional — Tailscale (cross-network access)

The above works on LAN only. For access from anywhere without router port forwarding: install [Tailscale](https://tailscale.com) on both machines, log in with the same account, and SSH to the Tailscale IP. Works through NAT, no portproxy or firewall rules needed. VSCode Remote-SSH works the same way — just use the Tailscale IP in your `config`.
