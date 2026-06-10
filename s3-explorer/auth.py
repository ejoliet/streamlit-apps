"""
User store for the S3 Explorer app.

Users live in users.yaml next to this file (auto-created on first run).
Passwords are hashed with PBKDF2-HMAC-SHA256. Roles: admin, write, read-only.
"""

import hashlib
import hmac
import os
import secrets
from typing import Optional

import yaml

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.yaml")

ROLES = ["admin", "write", "read-only"]

# Bootstrap accounts created on first run. Change these passwords immediately
# via the admin panel.
DEFAULT_ACCOUNTS = [
    ("admin", "admin", "admin"),
    ("roman", "roman", "read-only"),
]

_PBKDF2_ITERATIONS = 200_000


def _hash_password(password: str, salt_hex: Optional[str] = None) -> tuple[str, str]:
    salt_hex = salt_hex or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), _PBKDF2_ITERATIONS
    ).hex()
    return salt_hex, digest


def _bootstrap() -> dict:
    users = {}
    for name, password, role in DEFAULT_ACCOUNTS:
        salt, digest = _hash_password(password)
        users[name] = {"role": role, "salt": salt, "hash": digest}
    save_users(users)
    return users


def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return _bootstrap()
    with open(USERS_FILE) as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("users", {})


def save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as fh:
        yaml.safe_dump({"users": users}, fh, default_flow_style=False)
    os.chmod(USERS_FILE, 0o600)


def authenticate(username: str, password: str) -> Optional[str]:
    """Return the user's role if credentials are valid, else None."""
    users = load_users()
    record = users.get(username)
    if not record:
        return None
    _, digest = _hash_password(password, record["salt"])
    if hmac.compare_digest(digest, record["hash"]):
        return record["role"]
    return None


def add_user(username: str, password: str, role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role}")
    users = load_users()
    if username in users:
        raise ValueError(f"User '{username}' already exists")
    salt, digest = _hash_password(password)
    users[username] = {"role": role, "salt": salt, "hash": digest}
    save_users(users)


def set_password(username: str, password: str) -> None:
    users = load_users()
    if username not in users:
        raise ValueError(f"Unknown user: {username}")
    salt, digest = _hash_password(password)
    users[username].update(salt=salt, hash=digest)
    save_users(users)


def set_role(username: str, role: str) -> None:
    if role not in ROLES:
        raise ValueError(f"Unknown role: {role}")
    users = load_users()
    if username not in users:
        raise ValueError(f"Unknown user: {username}")
    if users[username]["role"] == "admin" and role != "admin" and _admin_count(users) == 1:
        raise ValueError("Cannot demote the last admin")
    users[username]["role"] = role
    save_users(users)


def delete_user(username: str) -> None:
    users = load_users()
    if username not in users:
        raise ValueError(f"Unknown user: {username}")
    if users[username]["role"] == "admin" and _admin_count(users) == 1:
        raise ValueError("Cannot delete the last admin")
    del users[username]
    save_users(users)


def _admin_count(users: dict) -> int:
    return sum(1 for u in users.values() if u["role"] == "admin")


if __name__ == "__main__":
    import argparse
    import getpass
    import sys

    parser = argparse.ArgumentParser(description="Manage S3 Explorer users")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List all users")

    p_passwd = sub.add_parser("passwd", help="Set a user's password")
    p_passwd.add_argument("username")
    p_passwd.add_argument("password", nargs="?", help="New password (prompted if omitted)")

    p_add = sub.add_parser("add", help="Add a new user")
    p_add.add_argument("username")
    p_add.add_argument("--role", choices=ROLES, default="read-only")
    p_add.add_argument("password", nargs="?", help="Password (prompted if omitted)")

    p_role = sub.add_parser("role", help="Change a user's role")
    p_role.add_argument("username")
    p_role.add_argument("role", choices=ROLES)

    p_del = sub.add_parser("delete", help="Delete a user")
    p_del.add_argument("username")

    args = parser.parse_args()

    if args.cmd == "list":
        users = load_users()
        if not users:
            print("No users found.")
        else:
            fmt = "{:<20} {}"
            print(fmt.format("USERNAME", "ROLE"))
            print("-" * 30)
            for name, rec in sorted(users.items()):
                print(fmt.format(name, rec["role"]))

    elif args.cmd == "passwd":
        pw = args.password or getpass.getpass(f"New password for {args.username}: ")
        try:
            set_password(args.username, pw)
            print(f"Password updated for '{args.username}'.")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "add":
        pw = args.password or getpass.getpass(f"Password for {args.username}: ")
        try:
            add_user(args.username, pw, args.role)
            print(f"Created user '{args.username}' with role '{args.role}'.")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "role":
        try:
            set_role(args.username, args.role)
            print(f"Role for '{args.username}' set to '{args.role}'.")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.cmd == "delete":
        try:
            delete_user(args.username)
            print(f"Deleted user '{args.username}'.")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
