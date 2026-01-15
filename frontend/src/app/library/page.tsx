"use client";

/**
 * Library Management Page
 * 
 * Allows users to:
 * - View all files in their library
 * - Upload new files
 * - Delete files
 * - Edit file metadata
 */

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  listLibraryFiles,
  uploadToLibrary,
  deleteLibraryFile,
  getLibraryPreviewUrl,
  getLibraryDownloadUrl,
  LibraryFile,
} from "@/lib/library-api";

export default function LibraryPage() {
  const [files, setFiles] = useState<LibraryFile[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filterType, setFilterType] = useState<"all" | "pdf" | "png">("all");
  
  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  
  // Selected file for details
  const [selectedFile, setSelectedFile] = useState<LibraryFile | null>(null);
  
  // Error state
  const [error, setError] = useState<string | null>(null);

  // Load library
  const loadLibrary = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await listLibraryFiles({
        file_type: filterType === "all" ? undefined : filterType,
        search: search || undefined,
      });
      setFiles(response.files);
    } catch (err) {
      console.error("Failed to load library:", err);
      setError("Failed to load library");
    } finally {
      setIsLoading(false);
    }
  }, [filterType, search]);

  useEffect(() => {
    loadLibrary();
  }, [loadLibrary]);

  // Handle file upload
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;

    setIsUploading(true);
    setError(null);

    try {
      for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        setUploadProgress(`Uploading ${file.name} (${i + 1}/${fileList.length})...`);
        await uploadToLibrary(file);
      }
      
      setUploadProgress("Upload complete!");
      await loadLibrary();
      
      setTimeout(() => {
        setUploadProgress("");
      }, 2000);
    } catch (err) {
      console.error("Upload failed:", err);
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  // Handle delete
  const handleDelete = async (fileId: string, fileName: string) => {
    if (!confirm(`Delete "${fileName}" from library?`)) return;

    try {
      await deleteLibraryFile(fileId);
      await loadLibrary();
      if (selectedFile?.id === fileId) {
        setSelectedFile(null);
      }
    } catch (err) {
      console.error("Delete failed:", err);
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  };

  // Filtered files
  const filteredFiles = files;

  return (
    <div className="min-h-screen bg-ink-950 flex flex-col">
      {/* Header */}
      <header className="h-14 bg-ink-900 border-b border-ink-700 flex items-center px-6 justify-between">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blueprint-500 to-blueprint-700 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h1 className="text-lg font-bold text-ink-100">File Library</h1>
        </Link>

        <Link
          href="/upload"
          className="px-4 py-2 bg-blueprint-600 hover:bg-blueprint-500 rounded-lg text-sm text-white"
        >
          Create Overlay
        </Link>
      </header>

      {/* Main content */}
      <main className="flex-1 flex">
        {/* File list */}
        <div className="flex-1 flex flex-col">
          {/* Toolbar */}
          <div className="p-4 border-b border-ink-800 flex items-center gap-4">
            {/* Search */}
            <input
              type="text"
              placeholder="Search files..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="flex-1 max-w-xs bg-ink-800 border border-ink-700 rounded-lg px-4 py-2 text-ink-200 placeholder-ink-500"
            />

            {/* Filter */}
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as "all" | "pdf" | "png")}
              className="bg-ink-800 border border-ink-700 rounded-lg px-4 py-2 text-ink-200"
            >
              <option value="all">All types</option>
              <option value="pdf">PDF only</option>
              <option value="png">PNG only</option>
            </select>

            {/* Upload button */}
            <label className="px-4 py-2 bg-blueprint-600 hover:bg-blueprint-500 rounded-lg text-sm text-white cursor-pointer">
              <input
                type="file"
                accept=".pdf,.png"
                multiple
                onChange={handleUpload}
                className="hidden"
                disabled={isUploading}
              />
              {isUploading ? uploadProgress : "Upload Files"}
            </label>
          </div>

          {/* Error message */}
          {error && (
            <div className="mx-4 mt-4 p-3 bg-red-500/10 border border-red-500/50 rounded-lg">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          {/* File grid */}
          <div className="flex-1 overflow-y-auto p-4">
            {isLoading ? (
              <div className="flex items-center justify-center h-64">
                <div className="w-8 h-8 border-4 border-blueprint-500 border-t-transparent rounded-full animate-spin" />
              </div>
            ) : filteredFiles.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-64 text-ink-500">
                <svg className="w-16 h-16 mb-4 text-ink-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                </svg>
                <p>No files in library</p>
                <p className="text-sm mt-1">Upload some files to get started</p>
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {filteredFiles.map((file) => (
                  <div
                    key={file.id}
                    onClick={() => setSelectedFile(file)}
                    className={`
                      p-4 bg-ink-900 border rounded-lg cursor-pointer transition-colors
                      ${selectedFile?.id === file.id
                        ? "border-blueprint-500"
                        : "border-ink-800 hover:border-ink-600"
                      }
                    `}
                  >
                    {/* Preview */}
                    <div className="aspect-square bg-ink-800 rounded mb-3 overflow-hidden">
                      {file.preview_path && (
                        <img
                          src={getLibraryPreviewUrl(file.id)}
                          alt={file.display_name}
                          className="w-full h-full object-contain"
                        />
                      )}
                    </div>

                    {/* Info */}
                    <p className="text-ink-200 font-medium truncate text-sm">
                      {file.display_name}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`
                        px-1.5 py-0.5 rounded text-xs uppercase
                        ${file.file_type === "pdf" ? "bg-red-500/20 text-red-400" : "bg-blue-500/20 text-blue-400"}
                      `}>
                        {file.file_type}
                      </span>
                      <span className="text-ink-600 text-xs">
                        {(file.size_bytes / 1024 / 1024).toFixed(1)} MB
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Status bar */}
          <div className="px-4 py-2 border-t border-ink-800 text-sm text-ink-500">
            {filteredFiles.length} files
          </div>
        </div>

        {/* Details panel */}
        {selectedFile && (
          <div className="w-80 bg-ink-900 border-l border-ink-700 flex flex-col">
            <div className="p-4 border-b border-ink-700 flex items-center justify-between">
              <h2 className="font-semibold text-ink-200">File Details</h2>
              <button
                onClick={() => setSelectedFile(null)}
                className="p-1 hover:bg-ink-800 rounded"
              >
                <svg className="w-4 h-4 text-ink-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4">
              {/* Preview */}
              <div className="aspect-square bg-ink-800 rounded-lg mb-4 overflow-hidden">
                {selectedFile.preview_path && (
                  <img
                    src={getLibraryPreviewUrl(selectedFile.id)}
                    alt={selectedFile.display_name}
                    className="w-full h-full object-contain"
                  />
                )}
              </div>

              {/* Metadata */}
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-ink-500 uppercase tracking-wider">Name</label>
                  <p className="text-ink-200">{selectedFile.display_name}</p>
                </div>

                <div>
                  <label className="text-xs text-ink-500 uppercase tracking-wider">Filename</label>
                  <p className="text-ink-400 text-sm break-all">{selectedFile.filename}</p>
                </div>

                <div className="flex gap-4">
                  <div>
                    <label className="text-xs text-ink-500 uppercase tracking-wider">Type</label>
                    <p className="text-ink-200 uppercase">{selectedFile.file_type}</p>
                  </div>
                  <div>
                    <label className="text-xs text-ink-500 uppercase tracking-wider">Size</label>
                    <p className="text-ink-200">{(selectedFile.size_bytes / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                </div>

                {selectedFile.preview_width && selectedFile.preview_height && (
                  <div>
                    <label className="text-xs text-ink-500 uppercase tracking-wider">Dimensions</label>
                    <p className="text-ink-200">{selectedFile.preview_width} × {selectedFile.preview_height} px</p>
                  </div>
                )}

                <div>
                  <label className="text-xs text-ink-500 uppercase tracking-wider">Uploaded</label>
                  <p className="text-ink-200">
                    {new Date(selectedFile.uploaded_at).toLocaleDateString()}
                  </p>
                </div>

                {selectedFile.tags.length > 0 && (
                  <div>
                    <label className="text-xs text-ink-500 uppercase tracking-wider">Tags</label>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {selectedFile.tags.map((tag) => (
                        <span key={tag} className="px-2 py-0.5 bg-ink-800 rounded text-xs text-ink-300">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {selectedFile.description && (
                  <div>
                    <label className="text-xs text-ink-500 uppercase tracking-wider">Description</label>
                    <p className="text-ink-400 text-sm">{selectedFile.description}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Actions */}
            <div className="p-4 border-t border-ink-700 space-y-2">
              <a
                href={getLibraryDownloadUrl(selectedFile.id)}
                download
                className="block w-full px-4 py-2 bg-ink-800 hover:bg-ink-700 rounded-lg text-sm text-ink-200 text-center"
              >
                Download Original
              </a>
              <button
                onClick={() => handleDelete(selectedFile.id, selectedFile.display_name)}
                className="w-full px-4 py-2 bg-red-500/20 hover:bg-red-500/30 border border-red-500/50 rounded-lg text-sm text-red-400"
              >
                Delete from Library
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
