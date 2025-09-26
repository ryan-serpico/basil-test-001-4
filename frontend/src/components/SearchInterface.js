import React, { useState } from 'react';
import axios from 'axios';
import SearchResults from './SearchResults';

const SearchInterface = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchInfo, setSearchInfo] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();

    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setResults([]);

      const response = await axios.post('/api/search', {
        query: query.trim(),
        limit: 20
      });

      setResults(response.data.results || []);
      setSearchInfo({
        query: response.data.query,
        totalResults: response.data.total_results
      });

    } catch (err) {
      console.error('Search error:', err);
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to perform search'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (format) => {
    if (!query.trim()) {
      alert('Please perform a search before exporting');
      return;
    }

    try {
      const response = await axios.post('/api/export', {
        query: query.trim(),
        format: format,
        limit: 100
      });

      if (format === 'json') {
        // Download JSON file
        const blob = new Blob([JSON.stringify(response.data, null, 2)], {
          type: 'application/json'
        });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `search_results_${query.replace(/\s+/g, '_')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } else if (format === 'csv') {
        // Download CSV file
        const blob = new Blob([response.data.data], {
          type: 'text/csv'
        });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `search_results_${query.replace(/\s+/g, '_')}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } else if (format === 'parquet') {
        // Handle Parquet export response
        alert(`Parquet export completed: ${response.data.message}\nInfo: ${response.data.note}`);
      }

    } catch (err) {
      console.error('Export error:', err);
      alert('Failed to export results: ' + (err.response?.data?.detail || err.message));
    }
  };

  const clearSearch = () => {
    setQuery('');
    setResults([]);
    setError(null);
    setSearchInfo(null);
  };

  return (
    <div className="search-interface">
      <div className="search-header">
        <h1>Search test-001 Documents</h1>
        <p>Use natural language to search through your document collection using semantic similarity</p>
      </div>

      <form onSubmit={handleSearch} className="search-form">
        <div className="search-input-container">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your search query..."
            className="search-input"
            disabled={loading}
          />
          <div className="search-buttons">
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="search-button"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
            {(results.length > 0 || query) && (
              <button
                type="button"
                onClick={clearSearch}
                className="clear-button"
                disabled={loading}
              >
                Clear
              </button>
            )}
          </div>
        </div>
      </form>

      {error && (
        <div className="error-message">
          <p>❌ {error}</p>
        </div>
      )}

      {searchInfo && (
        <div className="search-info">
          <p>
            Found <strong>{searchInfo.totalResults}</strong> results for "{searchInfo.query}"
          </p>
          {results.length > 0 && (
            <div className="export-options">
              <span>Export results: </span>
              <button
                onClick={() => handleExport('json')}
                className="export-button"
                disabled={loading}
              >
                JSON
              </button>
              <button
                onClick={() => handleExport('csv')}
                className="export-button"
                disabled={loading}
              >
                CSV
              </button>
              <button
                onClick={() => handleExport('parquet')}
                className="export-button"
                disabled={loading}
                title="Export full dataset in Parquet format"
              >
                Parquet
              </button>
            </div>
          )}
        </div>
      )}

      <SearchResults results={results} loading={loading} />

      {results.length === 0 && searchInfo && !loading && (
        <div className="no-results">
          <h3>No results found</h3>
          <p>Try rephrasing your search query or using different keywords.</p>
        </div>
      )}
    </div>
  );
};

export default SearchInterface;