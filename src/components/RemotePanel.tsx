import { useCallback, useEffect, useState } from "react";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { Icon } from "@iconify/react";
import { api } from "../lib/ipc";
import type { RemoteDevice, RemoteStatus } from "../types";

type Props = {
  initialStatus?: RemoteStatus | null;
  onStatusChange?: (status: RemoteStatus) => void;
};

const REMOTE_STATUS_EVENT = "remote-status-changed";

export function RemotePanel({ initialStatus = null, onStatusChange }: Props) {
  const [status, setStatus] = useState<RemoteStatus | null>(initialStatus);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const applyStatus = useCallback(
    (next: RemoteStatus) => {
      setStatus(next);
      onStatusChange?.(next);
    },
    [onStatusChange],
  );

  const refresh = useCallback(async () => {
    try {
      applyStatus(await api.remoteGetStatus());
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }, [applyStatus]);

  useEffect(() => {
    if (initialStatus) setStatus(initialStatus);
  }, [initialStatus]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    let cancelled = false;
    let unlisten: UnlistenFn | null = null;
    void listen<RemoteStatus>(REMOTE_STATUS_EVENT, (event) => {
      applyStatus(event.payload);
    }).then((nextUnlisten) => {
      if (cancelled) nextUnlisten();
      else unlisten = nextUnlisten;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [applyStatus]);

  const setEnabled = useCallback(async (enabled: boolean) => {
    setBusy(true);
    try {
      applyStatus(await api.remoteSetEnabled(enabled));
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }, [applyStatus]);

  const startPairing = useCallback(async () => {
    setBusy(true);
    try {
      applyStatus(await api.remoteStartPairing());
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }, [applyStatus]);

  const stopPairing = useCallback(async () => {
    setBusy(true);
    try {
      applyStatus(await api.remoteStopPairing());
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }, [applyStatus]);

  const revokeDevice = useCallback(async (device: RemoteDevice) => {
    if (!window.confirm(`Revoke ${device.name}? This device will lose access immediately.`)) {
      return;
    }
    setBusy(true);
    try {
      applyStatus(await api.remoteRevokeDevice(device.id));
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }, [applyStatus]);

  const pairing = status?.pairing ?? null;
  const activeDevices = status?.devices.filter((device) => !device.revokedAtMs) ?? [];
  const revokedDevices = status?.devices.filter((device) => device.revokedAtMs) ?? [];

  return (
    <div className="remote-panel">
      <section className="remote-panel__hero">
        <div>
          <p className="remote-panel__eyebrow">Remote</p>
          <h1>Control Sinew from your phone</h1>
          <p>
            The phone PWA connects through remote.sinew-ide.com while content stays
            end-to-end encrypted between this PC and paired devices.
          </p>
        </div>
        <button
          type="button"
          className="settings-pane__switch remote-panel__main-switch"
          role="switch"
          aria-checked={status?.enabled ?? false}
          data-on={status?.enabled ? "true" : "false"}
          disabled={busy || !status}
          onClick={() => setEnabled(!(status?.enabled ?? false))}
          title={status?.enabled ? "Disable Remote" : "Enable Remote"}
        >
          <span className="settings-pane__switch-thumb" />
        </button>
      </section>

      <section className="remote-panel__grid">
        <InfoCard
          icon="solar:cloud-check-linear"
          label="Relay"
          value={status?.relayConnected ? "Connected" : "Offline"}
          tone={status?.relayConnected ? "ok" : "muted"}
        />
        <InfoCard
          icon="solar:smartphone-2-linear"
          label="Devices online"
          value={`${activeDevices.filter((device) => device.connected).length}/${activeDevices.length}`}
          tone={activeDevices.some((device) => device.connected) ? "ok" : "muted"}
        />
        <InfoCard
          icon="solar:shield-keyhole-linear"
          label="PC identity"
          value={status?.pcId ?? "—"}
          tone="muted"
        />
      </section>

      <section className="remote-panel__section">
        <div className="remote-panel__section-head">
          <div>
            <h2>Pairing</h2>
            <p>The 6-digit code is accepted only while this screen is open.</p>
          </div>
          <button
            type="button"
            className="settings-pane__btn"
            data-primary={!pairing ? "true" : "false"}
            disabled={busy || !status?.enabled}
            onClick={pairing ? stopPairing : startPairing}
          >
            {pairing ? "Close pairing" : "Open pairing"}
          </button>
        </div>
        {pairing ? (
          <div className="remote-panel__pairing">
            <div className="remote-panel__code" aria-label="Pairing code">
              {pairing.code.split("").map((digit, index) => (
                <span key={`${digit}-${index}`}>{digit}</span>
              ))}
            </div>
            <QrCode value={pairing.qrUrl} />
            <div className="remote-panel__pairing-copy">
              <span>Scan the QR or open:</span>
              <a href={pairing.qrUrl}>{pairing.qrUrl}</a>
              <small>{pairing.attemptsRemaining} attempts remaining</small>
            </div>
          </div>
        ) : (
          <div className="remote-panel__empty">Pairing is closed.</div>
        )}
      </section>

      <section className="remote-panel__section">
        <div className="remote-panel__section-head">
          <div>
            <h2>Paired devices</h2>
            <p>Revoke a device immediately if a phone is lost or replaced.</p>
          </div>
        </div>
        <div className="remote-panel__devices">
          {activeDevices.length === 0 ? (
            <div className="remote-panel__empty">No paired devices yet.</div>
          ) : (
            activeDevices.map((device) => (
              <DeviceRow
                key={device.id}
                device={device}
                busy={busy}
                onRevoke={() => revokeDevice(device)}
              />
            ))
          )}
        </div>
        {revokedDevices.length > 0 && (
          <details className="remote-panel__revoked">
            <summary>{revokedDevices.length} revoked</summary>
            {revokedDevices.map((device) => (
              <DeviceRow key={device.id} device={device} busy={true} revoked />
            ))}
          </details>
        )}
      </section>

      {error && (
        <button type="button" className="remote-panel__error" onClick={() => setError(null)}>
          {error}
        </button>
      )}
    </div>
  );
}

function InfoCard({
  icon,
  label,
  value,
  tone,
}: {
  icon: string;
  label: string;
  value: string;
  tone: "ok" | "muted";
}) {
  return (
    <div className="remote-panel__info" data-tone={tone}>
      <Icon icon={icon} width={17} height={17} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DeviceRow({
  device,
  busy,
  revoked = false,
  onRevoke,
}: {
  device: RemoteDevice;
  busy: boolean;
  revoked?: boolean;
  onRevoke?: () => void;
}) {
  return (
    <div
      className="remote-panel__device"
      data-revoked={revoked ? "true" : "false"}
      data-connected={!revoked && device.connected ? "true" : "false"}
    >
      <div className="remote-panel__device-main">
        <span className="remote-panel__device-icon">
          <Icon icon="solar:smartphone-2-linear" width={16} height={16} />
        </span>
        <div className="remote-panel__device-text">
          <strong>{device.name}</strong>
          <span className="remote-panel__device-status">
            {revoked
              ? "Revoked"
              : device.connected
                ? "Connected now"
                : device.lastSeenAtMs
                  ? `Last seen ${formatDate(device.lastSeenAtMs)}`
                  : "Not seen yet"}
          </span>
        </div>
      </div>
      <div className="remote-panel__device-actions">
        {device.pushEnabled && !revoked && (
          <span className="remote-panel__device-push">
            <Icon icon="solar:bell-linear" width={12} height={12} />
            Push
          </span>
        )}
        {!revoked && (
          <button type="button" className="settings-pane__btn" disabled={busy} onClick={onRevoke}>
            Revoke
          </button>
        )}
      </div>
    </div>
  );
}

function QrCode({ value }: { value: string }) {
  const src = `https://api.qrserver.com/v1/create-qr-code/?size=180x180&margin=8&data=${encodeURIComponent(value)}`;
  return <img className="remote-panel__qr" src={src} alt="Remote pairing QR code" />;
}

function formatDate(ms: number) {
  try {
    return new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(ms));
  } catch {
    return new Date(ms).toLocaleString();
  }
}
