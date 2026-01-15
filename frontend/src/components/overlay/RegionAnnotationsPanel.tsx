"use client";

/**
 * Region Annotations Panel Component
 * 
 * Displays a list of detected difference regions as toggleable layers.
 * Each region can be shown/hidden individually, with a "Show All" / "Hide All" toggle.
 */

import React, { useMemo } from "react";
import { RegionInfo } from "@/lib/overlay-api";
import { RegionVisibilityState } from "@/lib/overlay-types";

interface RegionAnnotationsPanelProps {
  regions: RegionInfo[];
  regionVisibility: RegionVisibilityState;
  allRegionsVisible: boolean;
  onToggleRegion: (regionId: number) => void;
  onToggleAll: () => void;
  /** Color used for Drawing A (for computing contrast) */
  colorA?: string;
  /** Color used for Drawing B (for computing contrast) */
  colorB?: string;
}

// Color name to hex mapping
const COLOR_NAME_TO_HEX: Record<string, string> = {
  red: "#ff0000",
  cyan: "#00ffff",
  blue: "#0000ff",
  green: "#00ff00",
  magenta: "#ff00ff",
  yellow: "#ffff00",
  black: "#000000",
  white: "#ffffff",
};

function getHexColor(color: string | undefined): string {
  if (!color) return "#ff0000";
  if (color.startsWith("#")) return color;
  return COLOR_NAME_TO_HEX[color.toLowerCase()] || "#ff0000";
}

function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
    : [255, 0, 0];
}

function rgbToHex(r: number, g: number, b: number): string {
  return `#${[r, g, b].map(x => x.toString(16).padStart(2, '0')).join('')}`;
}

function computeContrastingColor(color1: string, color2: string): string {
  const c1 = hexToRgb(getHexColor(color1));
  const c2 = hexToRgb(getHexColor(color2));
  
  const candidates: [number, number, number][] = [
    [255, 255, 0], [0, 255, 0], [255, 165, 0], [255, 0, 255],
    [128, 0, 128], [0, 128, 0], [255, 255, 255], [0, 0, 0],
  ];
  
  const colorDistance = (a: [number, number, number], b: [number, number, number]): number =>
    Math.sqrt(a.reduce((sum, v, i) => sum + (v - b[i]) ** 2, 0));
  
  let best = candidates[0];
  let bestScore = 0;
  
  for (const c of candidates) {
    const score = Math.min(colorDistance(c, c1), colorDistance(c, c2));
    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }
  
  return rgbToHex(best[0], best[1], best[2]);
}

export function RegionAnnotationsPanel({
  regions,
  regionVisibility,
  allRegionsVisible,
  onToggleRegion,
  onToggleAll,
  colorA = "red",
  colorB = "cyan",
}: RegionAnnotationsPanelProps) {
  // Compute contrasting color for region badges
  const badgeColor = useMemo(
    () => computeContrastingColor(colorA, colorB),
    [colorA, colorB]
  );

  if (regions.length === 0) {
    return null;
  }

  return (
    <div className="border-t border-ink-700">
      {/* Header */}
      <div className="px-4 py-3 flex items-center justify-between border-b border-ink-800">
        <h3 className="text-sm font-semibold text-ink-200 uppercase tracking-wider">
          Region Annotations
        </h3>
        <button
          onClick={onToggleAll}
          className={`
            px-2 py-1 text-xs rounded font-medium
            ${allRegionsVisible 
              ? "bg-blueprint-600 text-white hover:bg-blueprint-500" 
              : "bg-ink-700 text-ink-300 hover:bg-ink-600"}
          `}
        >
          {allRegionsVisible ? "Hide All" : "Show All"}
        </button>
      </div>

      {/* Region List */}
      <div className="max-h-48 overflow-y-auto">
        {regions.map((region) => {
          const isVisible = regionVisibility[region.id] ?? true;
          
          return (
            <div
              key={region.id}
              className="flex items-center gap-3 px-4 py-2 border-b border-ink-800 hover:bg-ink-850 cursor-pointer"
              onClick={() => onToggleRegion(region.id)}
            >
              {/* Visibility toggle */}
              <button
                className={`p-1 rounded hover:bg-ink-700 ${
                  isVisible ? "text-ink-300" : "text-ink-600"
                }`}
                title={isVisible ? "Hide region" : "Show region"}
              >
                {isVisible ? (
                  <EyeIcon className="w-4 h-4" />
                ) : (
                  <EyeSlashIcon className="w-4 h-4" />
                )}
              </button>

              {/* Region label */}
              <div className="flex items-center gap-2 flex-1">
                <span
                  className="inline-flex items-center justify-center w-6 h-6 rounded text-xs font-bold text-white"
                  style={{
                    backgroundColor: isVisible ? badgeColor : "#374151",
                  }}
                >
                  R{region.id}
                </span>
                <span className={`text-sm ${isVisible ? "text-ink-300" : "text-ink-500"}`}>
                  Region {region.id}
                </span>
              </div>

              {/* Area indicator */}
              <span className="text-xs text-ink-500">
                {region.area.toLocaleString()} px
              </span>
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="px-4 py-2 text-xs text-ink-500">
        {regions.filter((r) => regionVisibility[r.id] ?? true).length} of {regions.length} visible
      </div>
    </div>
  );
}

// ============================================================
// Icon Components
// ============================================================

function EyeIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
      />
    </svg>
  );
}

function EyeSlashIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"
      />
    </svg>
  );
}
