"use client";

/**
 * Toolbar Component
 * 
 * Top toolbar with:
 * - Undo/Redo
 * - Zoom controls
 * - Export options
 * - Status indicator
 */

import React from "react";
import { ComposeStatus } from "@/lib/overlay-types";

interface ToolbarProps {
  sessionId: string;
  status: ComposeStatus;
  confidence: number;
  warnings: string[];
  zoom: number;
  canUndo: boolean;
  canRedo: boolean;
  isDirty: boolean;
  isExporting: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onZoomReset: () => void;
  onExportPNG: () => void;
  onExportState: () => void;
  onResetAll: () => void;
}

export function Toolbar({
  sessionId,
  status,
  confidence,
  warnings,
  zoom,
  canUndo,
  canRedo,
  isDirty,
  isExporting,
  onUndo,
  onRedo,
  onZoomIn,
  onZoomOut,
  onZoomReset,
  onExportPNG,
  onExportState,
  onResetAll,
}: ToolbarProps) {
  const isAutoFailed = status === "auto_failed_fallback_manual";

  return (
    <div className="h-12 bg-ink-900 border-b border-ink-700 flex items-center px-4 gap-4">
      {/* Status indicator */}
      <div className="flex items-center gap-2">
        <div
          className={`w-2 h-2 rounded-full ${
            isAutoFailed ? "bg-yellow-500" : "bg-green-500"
          }`}
        />
        <span className="text-sm text-ink-300">
          {isAutoFailed ? "Manual mode" : `Auto (${Math.round(confidence * 100)}%)`}
        </span>
        {isDirty && (
          <span className="text-xs text-yellow-500">• Unsaved changes</span>
        )}
      </div>

      {/* Warning badge */}
      {warnings.length > 0 && (
        <div className="relative group">
          <div className="px-2 py-1 bg-yellow-500/20 border border-yellow-500/50 rounded text-xs text-yellow-400 cursor-help">
            {warnings.length} warning{warnings.length > 1 ? "s" : ""}
          </div>
          
          {/* Tooltip */}
          <div className="absolute top-full left-0 mt-2 w-64 p-3 bg-ink-800 border border-ink-600 rounded shadow-xl hidden group-hover:block z-50">
            <ul className="text-xs text-ink-300 space-y-1">
              {warnings.map((w, i) => (
                <li key={i}>• {w}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Undo/Redo */}
      <div className="flex items-center gap-1">
        <button
          onClick={onUndo}
          disabled={!canUndo}
          className={`p-2 rounded ${
            canUndo
              ? "hover:bg-ink-700 text-ink-300"
              : "text-ink-600 cursor-not-allowed"
          }`}
          title="Undo (Ctrl+Z)"
        >
          <UndoIcon className="w-4 h-4" />
        </button>
        <button
          onClick={onRedo}
          disabled={!canRedo}
          className={`p-2 rounded ${
            canRedo
              ? "hover:bg-ink-700 text-ink-300"
              : "text-ink-600 cursor-not-allowed"
          }`}
          title="Redo (Ctrl+Shift+Z)"
        >
          <RedoIcon className="w-4 h-4" />
        </button>
      </div>

      {/* Divider */}
      <div className="w-px h-6 bg-ink-700" />

      {/* Zoom controls */}
      <div className="flex items-center gap-1">
        <button
          onClick={onZoomOut}
          className="p-2 rounded hover:bg-ink-700 text-ink-300"
          title="Zoom out"
        >
          <MinusIcon className="w-4 h-4" />
        </button>
        <button
          onClick={onZoomReset}
          className="px-2 py-1 min-w-[60px] rounded hover:bg-ink-700 text-sm text-ink-300 text-center"
          title="Reset zoom"
        >
          {Math.round(zoom * 100)}%
        </button>
        <button
          onClick={onZoomIn}
          className="p-2 rounded hover:bg-ink-700 text-ink-300"
          title="Zoom in"
        >
          <PlusIcon className="w-4 h-4" />
        </button>
      </div>

      {/* Divider */}
      <div className="w-px h-6 bg-ink-700" />

      {/* Reset button */}
      <button
        onClick={onResetAll}
        className="px-3 py-1.5 rounded hover:bg-ink-700 text-sm text-ink-300"
        title="Reset all layers to auto-aligned positions"
      >
        Reset All
      </button>

      {/* Divider */}
      <div className="w-px h-6 bg-ink-700" />

      {/* Export buttons */}
      <div className="flex items-center gap-2">
        <button
          onClick={onExportState}
          className="px-3 py-1.5 bg-ink-800 hover:bg-ink-700 rounded text-sm text-ink-300 border border-ink-600"
          disabled={isExporting}
        >
          Export State
        </button>
        <button
          onClick={onExportPNG}
          className="px-3 py-1.5 bg-blueprint-600 hover:bg-blueprint-500 rounded text-sm text-white"
          disabled={isExporting}
        >
          {isExporting ? "Exporting..." : "Export PNG"}
        </button>
      </div>
    </div>
  );
}

// ============================================================
// Icon Components
// ============================================================

function UndoIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6"
      />
    </svg>
  );
}

function RedoIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M21 10h-10a8 8 0 00-8 8v2M21 10l-6 6m6-6l-6-6"
      />
    </svg>
  );
}

function PlusIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M12 6v6m0 0v6m0-6h6m-6 0H6"
      />
    </svg>
  );
}

function MinusIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M20 12H4"
      />
    </svg>
  );
}
