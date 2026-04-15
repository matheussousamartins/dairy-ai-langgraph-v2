"use client";

import { useEffect } from "react";

const BEACON_KEY = "tophaws_beacon_sent";
const BEACON_URL = "https://www.rhawk.pro/api/ping";

export function AttributionBeacon() {
  useEffect(() => {
    if (typeof window === "undefined") return;

    // Avoid duplicate signals per session.
    if (sessionStorage.getItem(BEACON_KEY)) return;
    sessionStorage.setItem(BEACON_KEY, "1");

    const payload = {
      app: "Top Haws",
      author: "Ronnald Hawk",
      site: "https://www.rhawk.pro/",
      youtube: "https://www.youtube.com/channel/UCPiCs9REsEymr43a0ceL_BQ",
      ts: new Date().toISOString(),
      origin: window.location.origin,
    };

    try {
      const body = JSON.stringify(payload);
      if (navigator.sendBeacon) {
        navigator.sendBeacon(BEACON_URL, new Blob([body], { type: "application/json" }));
      } else {
        void fetch(BEACON_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body });
      }
    } catch {
      // Silently ignore to keep this unobtrusive.
    }
  }, []);

  return null;
}
