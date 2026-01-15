import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-ink-950 flex flex-col">
      {/* Header */}
      <header className="h-14 bg-ink-900 border-b border-ink-700 flex items-center px-6">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blueprint-500 to-blueprint-700 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h1 className="text-lg font-bold text-ink-100">Archiboost</h1>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-4xl">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-ink-100 mb-4">
              Architectural Detail Comparison
            </h1>
            <p className="text-xl text-ink-400">
              Compare, align, and overlay architectural drawings
            </p>
          </div>

          <div className="grid grid-cols-2 gap-6 max-w-2xl mx-auto">
            {/* Upload/Overlay */}
            <Link
              href="/upload"
              className="p-6 bg-ink-900 border border-ink-700 rounded-xl hover:border-blueprint-500 transition-colors group"
            >
              <div className="w-14 h-14 mx-auto mb-4 rounded-xl bg-gradient-to-br from-blueprint-500 to-blueprint-700 flex items-center justify-center group-hover:scale-110 transition-transform">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <h2 className="text-lg font-semibold text-ink-100 text-center mb-2">
                Create Overlay
              </h2>
              <p className="text-ink-400 text-center text-sm">
                Create aligned overlay from library or new uploads
              </p>
            </Link>

            {/* File Library */}
            <Link
              href="/library"
              className="p-6 bg-ink-900 border border-ink-700 rounded-xl hover:border-blueprint-500 transition-colors group"
            >
              <div className="w-14 h-14 mx-auto mb-4 rounded-xl bg-gradient-to-br from-blueprint-500 to-blueprint-700 flex items-center justify-center group-hover:scale-110 transition-transform">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                </svg>
              </div>
              <h2 className="text-lg font-semibold text-ink-100 text-center mb-2">
                File Library
              </h2>
              <p className="text-ink-400 text-center text-sm">
                Manage your uploaded files for easy reuse
              </p>
            </Link>
          </div>

          {/* Features */}
          <div className="mt-16 grid grid-cols-3 gap-6">
            <div className="text-center">
              <div className="w-12 h-12 mx-auto mb-4 rounded-lg bg-ink-800 flex items-center justify-center">
                <svg className="w-6 h-6 text-blueprint-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
                </svg>
              </div>
              <h3 className="font-semibold text-ink-200 mb-1">Layer-Based Editing</h3>
              <p className="text-sm text-ink-500">
                Non-destructive editing with full layer control
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-12 h-12 mx-auto mb-4 rounded-lg bg-ink-800 flex items-center justify-center">
                <svg className="w-6 h-6 text-blueprint-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <h3 className="font-semibold text-ink-200 mb-1">Auto Alignment</h3>
              <p className="text-sm text-ink-500">
                Automatic feature matching with manual fallback
              </p>
            </div>
            
            <div className="text-center">
              <div className="w-12 h-12 mx-auto mb-4 rounded-lg bg-ink-800 flex items-center justify-center">
                <svg className="w-6 h-6 text-blueprint-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
              </div>
              <h3 className="font-semibold text-ink-200 mb-1">Export Options</h3>
              <p className="text-sm text-ink-500">
                Export PNG or save state for later editing
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-4 text-center text-ink-600 text-sm">
        Archiboost Architectural Detail Comparison Tool
      </footer>
    </div>
  );
}
