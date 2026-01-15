/**
 * Library API Client
 * 
 * Client for the persistent file library.
 */

import { z } from "zod";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

// ============================================================
// Types
// ============================================================

export interface LibraryFile {
  id: string;
  filename: string;
  display_name: string;
  file_type: "pdf" | "png";
  size_bytes: number;
  storage_path: string;
  preview_path: string | null;
  preview_width: number | null;
  preview_height: number | null;
  uploaded_at: string;
  tags: string[];
  description: string | null;
  source: string;
}

export interface LibraryListResponse {
  files: LibraryFile[];
  total: number;
}

export interface LibraryUploadResponse {
  file: LibraryFile;
  message: string;
}

export interface CreateSessionResponse {
  session_id: string;
  file_a_source: string;
  file_b_source: string;
  message: string;
}

// ============================================================
// Schemas
// ============================================================

const LibraryFileSchema = z.object({
  id: z.string(),
  filename: z.string(),
  display_name: z.string(),
  file_type: z.enum(["pdf", "png"]),
  size_bytes: z.number(),
  storage_path: z.string(),
  preview_path: z.string().nullable(),
  preview_width: z.number().nullable(),
  preview_height: z.number().nullable(),
  uploaded_at: z.string(),
  tags: z.array(z.string()),
  description: z.string().nullable(),
  source: z.string(),
});

const LibraryListResponseSchema = z.object({
  files: z.array(LibraryFileSchema),
  total: z.number(),
});

// ============================================================
// API Error
// ============================================================

export class LibraryApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: unknown
  ) {
    super(message);
    this.name = "LibraryApiError";
  }
}

// ============================================================
// API Functions
// ============================================================

/**
 * List files in the library.
 */
export async function listLibraryFiles(options?: {
  file_type?: "pdf" | "png";
  tags?: string;
  search?: string;
}): Promise<LibraryListResponse> {
  const params = new URLSearchParams();
  if (options?.file_type) params.set("file_type", options.file_type);
  if (options?.tags) params.set("tags", options.tags);
  if (options?.search) params.set("search", options.search);

  const url = `${API_BASE_URL}/library/files${params.toString() ? `?${params}` : ""}`;
  
  const response = await fetch(url);

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new LibraryApiError(
      response.status,
      errorData.detail?.code || "LIST_FAILED",
      errorData.detail?.message || "Failed to list library files"
    );
  }

  const data = await response.json();
  const parsed = LibraryListResponseSchema.safeParse(data);
  
  if (!parsed.success) {
    console.warn("Library list validation failed:", parsed.error);
    return data as LibraryListResponse;
  }
  
  return parsed.data;
}

/**
 * Upload a file to the library.
 */
export async function uploadToLibrary(
  file: File,
  options?: {
    display_name?: string;
    tags?: string;
    description?: string;
  }
): Promise<LibraryUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const params = new URLSearchParams();
  if (options?.display_name) params.set("display_name", options.display_name);
  if (options?.tags) params.set("tags", options.tags);
  if (options?.description) params.set("description", options.description);

  const url = `${API_BASE_URL}/library/upload${params.toString() ? `?${params}` : ""}`;

  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new LibraryApiError(
      response.status,
      errorData.detail?.code || "UPLOAD_FAILED",
      errorData.detail?.message || "Failed to upload file"
    );
  }

  return response.json();
}

/**
 * Delete a file from the library.
 */
export async function deleteLibraryFile(fileId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/library/files/${fileId}`, {
    method: "DELETE",
  });

  if (!response.ok && response.status !== 204) {
    const errorData = await response.json().catch(() => ({}));
    throw new LibraryApiError(
      response.status,
      errorData.detail?.code || "DELETE_FAILED",
      errorData.detail?.message || "Failed to delete file"
    );
  }
}

/**
 * Create a session from library files and/or uploads.
 */
export async function createSessionFromLibrary(options: {
  file_a_id?: string;
  file_b_id?: string;
  file_a?: File;
  file_b?: File;
}): Promise<CreateSessionResponse> {
  const formData = new FormData();
  
  if (options.file_a) {
    formData.append("file_a", options.file_a);
  }
  if (options.file_b) {
    formData.append("file_b", options.file_b);
  }

  const params = new URLSearchParams();
  if (options.file_a_id) params.set("file_a_id", options.file_a_id);
  if (options.file_b_id) params.set("file_b_id", options.file_b_id);

  const url = `${API_BASE_URL}/library/create-session${params.toString() ? `?${params}` : ""}`;

  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new LibraryApiError(
      response.status,
      errorData.detail?.code || "SESSION_FAILED",
      errorData.detail?.message || "Failed to create session"
    );
  }

  return response.json();
}

/**
 * Get the preview URL for a library file.
 */
export function getLibraryPreviewUrl(fileId: string): string {
  return `${API_BASE_URL}/library/files/${fileId}/preview`;
}

/**
 * Get the download URL for a library file.
 */
export function getLibraryDownloadUrl(fileId: string): string {
  return `${API_BASE_URL}/library/files/${fileId}/download`;
}
