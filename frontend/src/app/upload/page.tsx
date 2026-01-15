"use client";

/**
 * Upload Page
 * 
 * Allows users to:
 * - Upload two files (PDF or PNG) for overlay comparison
 * - Select files from the library
 * - Upload new files to the library
 * - Mix library and fresh uploads
 */

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { generatePreview } from "@/lib/overlay-api";
import {
  listLibraryFiles,
  uploadToLibrary,
  createSessionFromLibrary,
  getLibraryPreviewUrl,
  LibraryFile,
} from "@/lib/library-api";

type FileSource = "upload" | "library";
type UploadState = "idle" | "uploading" | "generating_previews" | "complete" | "error";

interface FileSelection {
  source: FileSource;
  file?: File;
  libraryFile?: LibraryFile;
}

export default function UploadPage() {
  const router = useRouter();
  
  // File selections
  const [selectionA, setSelectionA] = useState<FileSelection | null>(null);
  const [selectionB, setSelectionB] = useState<FileSelection | null>(null);
  
  // Library state
  const [libraryFiles, setLibraryFiles] = useState<LibraryFile[]>([]);
  const [isLoadingLibrary, setIsLoadingLibrary] = useState(true);
  const [showLibraryPicker, setShowLibraryPicker] = useState<"A" | "B" | null>(null);
  const [librarySearch, setLibrarySearch] = useState("");
  
  // Upload to library toggle
  const [addToLibrary, setAddToLibrary] = useState(false);
  
  // Processing state
  const [state, setState] = useState<UploadState>("idle");
  const [progress, setProgress] = useState("");
  const [error, setError] = useState<string | null>(null);

  // Load library files
  useEffect(() => {
    async function loadLibrary() {
      try {
        setIsLoadingLibrary(true);
        const response = await listLibraryFiles();
        setLibraryFiles(response.files);
      } catch (err) {
        console.error("Failed to load library:", err);
      } finally {
        setIsLoadingLibrary(false);
      }
    }
    loadLibrary();
  }, []);

  // Handle file drop/select
  const handleFileChange = (which: "A" | "B") => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const selection: FileSelection = { source: "upload", file };
      if (which === "A") setSelectionA(selection);
      else setSelectionB(selection);
    }
  };

  const handleDrop = (which: "A" | "B") => (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const selection: FileSelection = { source: "upload", file };
      if (which === "A") setSelectionA(selection);
      else setSelectionB(selection);
    }
  };

  // Handle library selection
  const handleSelectFromLibrary = (which: "A" | "B", libraryFile: LibraryFile) => {
    const selection: FileSelection = { source: "library", libraryFile };
    if (which === "A") setSelectionA(selection);
    else setSelectionB(selection);
    setShowLibraryPicker(null);
  };

  // Clear selection
  const clearSelection = (which: "A" | "B") => {
    if (which === "A") setSelectionA(null);
    else setSelectionB(null);
  };

  // Handle submit
  const handleSubmit = useCallback(async () => {
    if (!selectionA || !selectionB) {
      setError("Please select both files");
      return;
    }

    try {
      setError(null);
      setState("uploading");
      setProgress("Creating session...");

      // If adding to library, upload files first
      if (addToLibrary) {
        if (selectionA.source === "upload" && selectionA.file) {
          setProgress("Uploading file A to library...");
          const result = await uploadToLibrary(selectionA.file);
          selectionA.libraryFile = result.file;
          selectionA.source = "library";
        }
        if (selectionB.source === "upload" && selectionB.file) {
          setProgress("Uploading file B to library...");
          const result = await uploadToLibrary(selectionB.file);
          selectionB.libraryFile = result.file;
          selectionB.source = "library";
        }
        // Refresh library
        const response = await listLibraryFiles();
        setLibraryFiles(response.files);
      }

      // Create session
      setProgress("Creating session...");
      const sessionResponse = await createSessionFromLibrary({
        file_a_id: selectionA.source === "library" ? selectionA.libraryFile?.id : undefined,
        file_b_id: selectionB.source === "library" ? selectionB.libraryFile?.id : undefined,
        file_a: selectionA.source === "upload" ? selectionA.file : undefined,
        file_b: selectionB.source === "upload" ? selectionB.file : undefined,
      });

      const sessionId = sessionResponse.session_id;

      // Generate previews
      setState("generating_previews");
      setProgress("Generating preview A...");
      await generatePreview(sessionId, "A");

      setProgress("Generating preview B...");
      await generatePreview(sessionId, "B");

      setState("complete");
      setProgress("Complete! Redirecting...");

      // Redirect to overlay editor
      setTimeout(() => {
        router.push(`/overlay?session=${sessionId}`);
      }, 500);

    } catch (err) {
      console.error("Upload failed:", err);
      setState("error");
      setError(err instanceof Error ? err.message : "Upload failed");
    }
  }, [selectionA, selectionB, addToLibrary, router]);

  const isProcessing = state === "uploading" || state === "generating_previews";

  // Filter library files by search
  const filteredLibraryFiles = libraryFiles.filter((f) =>
    librarySearch
      ? f.display_name.toLowerCase().includes(librarySearch.toLowerCase()) ||
        f.filename.toLowerCase().includes(librarySearch.toLowerCase())
      : true
  );

  return (
    <div className="min-h-screen bg-ink-950 flex flex-col">
      {/* Header */}
      <header className="h-14 bg-ink-900 border-b border-ink-700 flex items-center px-6">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blueprint-500 to-blueprint-700 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h1 className="text-lg font-bold text-ink-100">Overlay Editor</h1>
        </Link>
      </header>

      {/* Main content */}
      <main className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-5xl">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-ink-100 mb-2">
              Create Overlay Comparison
            </h1>
            <p className="text-ink-400">
              Select files from your library or upload new ones
            </p>
          </div>

          <div className="grid grid-cols-2 gap-8 mb-8">
            {/* File A */}
            <FileSelector
              label="A"
              color="red"
              selection={selectionA}
              onFileSelect={handleFileChange("A")}
              onFileDrop={handleDrop("A")}
              onSelectFromLibrary={() => setShowLibraryPicker("A")}
              onClear={() => clearSelection("A")}
            />

            {/* File B */}
            <FileSelector
              label="B"
              color="cyan"
              selection={selectionB}
              onFileSelect={handleFileChange("B")}
              onFileDrop={handleDrop("B")}
              onSelectFromLibrary={() => setShowLibraryPicker("B")}
              onClear={() => clearSelection("B")}
            />
          </div>

          {/* Add to library toggle */}
          <div className="flex items-center justify-center gap-3 mb-8">
            <input
              type="checkbox"
              id="addToLibrary"
              checked={addToLibrary}
              onChange={(e) => setAddToLibrary(e.target.checked)}
              className="w-4 h-4 bg-ink-800 border-ink-700 rounded"
            />
            <label htmlFor="addToLibrary" className="text-ink-300">
              Save uploaded files to library for future use
            </label>
          </div>

          {/* Submit button */}
          <div className="text-center">
            <button
              onClick={handleSubmit}
              disabled={!selectionA || !selectionB || isProcessing}
              className={`
                px-8 py-3 rounded-lg font-semibold text-white
                transition-colors
                ${!selectionA || !selectionB || isProcessing
                  ? "bg-ink-700 cursor-not-allowed"
                  : "bg-blueprint-600 hover:bg-blueprint-500"
                }
              `}
            >
              {isProcessing ? (
                <span className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  {progress}
                </span>
              ) : (
                "Create Overlay"
              )}
            </button>
          </div>

          {/* Error message */}
          {error && (
            <div className="mt-6 p-4 bg-red-500/10 border border-red-500/50 rounded-lg text-center">
              <p className="text-red-400">{error}</p>
            </div>
          )}

          {/* Library info */}
          <div className="mt-8 text-center text-ink-500 text-sm">
            {isLoadingLibrary ? (
              "Loading library..."
            ) : (
              `${libraryFiles.length} files in your library`
            )}
          </div>
        </div>
      </main>

      {/* Library Picker Modal */}
      {showLibraryPicker && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-8">
          <div className="bg-ink-900 rounded-xl border border-ink-700 w-full max-w-3xl max-h-[80vh] flex flex-col">
            {/* Modal header */}
            <div className="p-4 border-b border-ink-700 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-ink-100">
                Select from Library for Drawing {showLibraryPicker}
              </h2>
              <button
                onClick={() => setShowLibraryPicker(null)}
                className="p-2 hover:bg-ink-800 rounded"
              >
                <svg className="w-5 h-5 text-ink-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Search */}
            <div className="p-4 border-b border-ink-800">
              <input
                type="text"
                placeholder="Search library..."
                value={librarySearch}
                onChange={(e) => setLibrarySearch(e.target.value)}
                className="w-full bg-ink-800 border border-ink-700 rounded-lg px-4 py-2 text-ink-200 placeholder-ink-500"
              />
            </div>

            {/* File list */}
            <div className="flex-1 overflow-y-auto p-4">
              {filteredLibraryFiles.length === 0 ? (
                <div className="text-center text-ink-500 py-8">
                  {libraryFiles.length === 0
                    ? "No files in library. Upload some files first!"
                    : "No files match your search"}
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-4">
                  {filteredLibraryFiles.map((file) => (
                    <button
                      key={file.id}
                      onClick={() => handleSelectFromLibrary(showLibraryPicker, file)}
                      className="p-4 bg-ink-800 hover:bg-ink-750 border border-ink-700 hover:border-blueprint-500 rounded-lg text-left transition-colors"
                    >
                      <div className="flex gap-3">
                        {/* Preview thumbnail */}
                        <div className="w-16 h-16 bg-ink-700 rounded flex-shrink-0 overflow-hidden">
                          {file.preview_path && (
                            <img
                              src={getLibraryPreviewUrl(file.id)}
                              alt={file.display_name}
                              className="w-full h-full object-cover"
                            />
                          )}
                        </div>
                        
                        {/* Info */}
                        <div className="flex-1 min-w-0">
                          <p className="text-ink-100 font-medium truncate">
                            {file.display_name}
                          </p>
                          <p className="text-ink-500 text-sm truncate">
                            {file.filename}
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className={`
                              px-2 py-0.5 rounded text-xs uppercase
                              ${file.file_type === "pdf" ? "bg-red-500/20 text-red-400" : "bg-blue-500/20 text-blue-400"}
                            `}>
                              {file.file_type}
                            </span>
                            <span className="text-ink-600 text-xs">
                              {(file.size_bytes / 1024 / 1024).toFixed(1)} MB
                            </span>
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================
// File Selector Component
// ============================================================

function FileSelector({
  label,
  color,
  selection,
  onFileSelect,
  onFileDrop,
  onSelectFromLibrary,
  onClear,
}: {
  label: "A" | "B";
  color: "red" | "cyan";
  selection: FileSelection | null;
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onFileDrop: (e: React.DragEvent) => void;
  onSelectFromLibrary: () => void;
  onClear: () => void;
}) {
  const colorClasses = {
    red: {
      border: "border-red-500/50",
      bg: "bg-red-500/10",
      text: "text-red-400",
      badge: "bg-red-500/20",
    },
    cyan: {
      border: "border-cyan-500/50",
      bg: "bg-cyan-500/10",
      text: "text-cyan-400",
      badge: "bg-cyan-500/20",
    },
  }[color];

  const hasSelection = selection !== null;
  const displayName = selection?.source === "library"
    ? selection.libraryFile?.display_name
    : selection?.file?.name;
  const fileSize = selection?.source === "library"
    ? selection.libraryFile?.size_bytes
    : selection?.file?.size;

  return (
    <div
      className={`
        relative p-6 border-2 border-dashed rounded-xl text-center
        transition-colors
        ${hasSelection
          ? `${colorClasses.border} ${colorClasses.bg}`
          : "border-ink-700 bg-ink-900/50"
        }
      `}
      onDrop={onFileDrop}
      onDragOver={(e) => e.preventDefault()}
    >
      <input
        id={`file-${label.toLowerCase()}`}
        type="file"
        accept=".pdf,.png"
        onChange={onFileSelect}
        className="hidden"
      />
      
      {/* Label badge */}
      <div className="mb-4">
        <div className={`
          w-12 h-12 mx-auto rounded-full flex items-center justify-center
          ${hasSelection ? colorClasses.badge : "bg-ink-800"}
        `}>
          <span className={`text-2xl font-bold ${colorClasses.text}`}>{label}</span>
        </div>
      </div>

      {hasSelection ? (
        <>
          {/* Source badge */}
          <div className="flex justify-center mb-2">
            <span className={`
              px-2 py-0.5 rounded text-xs uppercase
              ${selection.source === "library" ? "bg-purple-500/20 text-purple-400" : "bg-green-500/20 text-green-400"}
            `}>
              {selection.source === "library" ? "From Library" : "New Upload"}
            </span>
          </div>
          
          <p className="text-ink-200 font-medium mb-1 truncate px-4">
            {displayName}
          </p>
          <p className="text-ink-500 text-sm mb-4">
            {fileSize ? `${(fileSize / 1024 / 1024).toFixed(2)} MB` : ""}
          </p>
          
          <button
            onClick={onClear}
            className="text-sm text-ink-500 hover:text-ink-300 underline"
          >
            Change selection
          </button>
        </>
      ) : (
        <>
          <p className="text-ink-300 mb-4">
            Choose Drawing {label}
          </p>
          
          <div className="flex flex-col gap-2">
            <button
              onClick={onSelectFromLibrary}
              className="px-4 py-2 bg-ink-800 hover:bg-ink-700 border border-ink-600 rounded-lg text-sm text-ink-200 transition-colors"
            >
              📚 Select from Library
            </button>
            
            <span className="text-ink-600 text-sm">or</span>
            
            <button
              onClick={() => document.getElementById(`file-${label.toLowerCase()}`)?.click()}
              className="px-4 py-2 bg-blueprint-600 hover:bg-blueprint-500 rounded-lg text-sm text-white transition-colors"
            >
              📤 Upload New File
            </button>
          </div>
          
          <p className="text-ink-600 text-xs mt-4">
            PDF or PNG (max 25MB)
          </p>
        </>
      )}
    </div>
  );
}
