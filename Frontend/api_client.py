import requests
import json
from typing import Dict, List, Optional, Any
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.timeout = 30
        
    def _make_request(self, method: str, endpoint: str, **kwargs):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            st.error(f"Backend connection error: {str(e)}")
            return None
    
    def health_check(self) -> bool:
        """Check if backend is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_build_cycles(self) -> List[Dict]:
        return self._make_request('GET', 'api/build-cycles') or []
    
    def get_dashboard_metrics(self, build_cycle: str = None, drift_phase: str = "Backend,Frontend") -> Dict:
        params = {}
        if build_cycle:
            params['build_cycle'] = build_cycle
        if drift_phase:
            params['drift_phase'] = drift_phase
        return self._make_request('GET', 'api/dashboard/metrics', params=params) or {}
    
    def get_risk_heatmap(self, build_cycle: str = None, drift_phase: str = "Backend,Frontend") -> Dict:
        params = {}
        if build_cycle:
            params['build_cycle'] = build_cycle
        if drift_phase:
            params['drift_phase'] = drift_phase
        return self._make_request('GET', 'api/risk-heatmap', params=params) or {}
    
    def get_test_stability(self, build_cycle: str = None, drift_phase: str = "Backend,Frontend") -> Dict:
        params = {}
        if build_cycle:
            params['build_cycle'] = build_cycle
        if drift_phase:
            params['drift_phase'] = drift_phase
        return self._make_request('GET', 'api/test-stability', params=params) or {}
    
    def get_recent_builds(self, limit: int = 10) -> List[Dict]:
        params = {'limit': limit}
        return self._make_request('GET', 'api/builds/recent', params=params) or []
    
    def get_test_distribution(self, build_cycle: str = None, drift_phase: str = "Backend,Frontend") -> Dict:
        params = {}
        if build_cycle:
            params['build_cycle'] = build_cycle
        if drift_phase:
            params['drift_phase'] = drift_phase
        return self._make_request('GET', 'api/tests/distribution', params=params) or {}
    
    def get_failure_analysis(self, build_cycle: str = None, drift_phase: str = "Backend,Frontend") -> Dict:
        params = {}
        if build_cycle:
            params['build_cycle'] = build_cycle
        if drift_phase:
            params['drift_phase'] = drift_phase
        return self._make_request('GET', 'api/failures/analysis', params=params) or {}
    
    def trigger_simulation(self) -> Dict:
        return self._make_request('POST', 'api/simulation/trigger') or {}
    
    def get_simulation_status(self) -> Dict:
        return self._make_request('GET', 'api/simulation/status') or {}
    
    def get_raw_data_sample(self, limit: int = 100) -> List[Dict]:
        params = {'limit': limit}
        return self._make_request('GET', 'api/raw-data/sample', params=params) or []
    
    def get_ai_predictions(self, limit: int = 20) -> List[Dict]:
        params = {'limit': limit}
        return self._make_request('GET', 'api/ai/predictions', params=params) or []
    
    def get_ai_recommendations(self, limit: int = 20) -> List[Dict]:
        """Get AI-recommended test execution order"""
        params = {'limit': limit}
        return self._make_request('GET', 'api/ai/recommendations', params=params) or []
    
    def get_high_risk_tests(self, threshold: float = 0.8) -> List[Dict]:
        """Get tests with high failure probability"""
        params = {'threshold': threshold}
        return self._make_request('GET', 'api/ai/high-risk-tests', params=params) or []
    
    def get_test_details(self, test_id: str) -> Dict:
        """Get detailed information about a specific test"""
        return self._make_request('GET', f'api/tests/{test_id}/details') or {}
    
    def search_tests(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search tests by ID or keyword"""
        params = {'query': query}
        if filters:
            params.update(filters)
        return self._make_request('GET', 'api/tests/search', params=params) or []
    
    def get_historical_metrics(self, start_date: str, end_date: str) -> Dict:
        """Get historical metrics for date range"""
        params = {'start_date': start_date, 'end_date': end_date}
        return self._make_request('GET', 'api/metrics/historical', params=params) or {}
    
    def export_report(self, format: str, filters: Dict = None) -> bytes:
        """Export report in specified format"""
        params = {'format': format}
        if filters:
            params.update(filters)
        try:
            url = f"{self.base_url}/api/export/report"
            response = requests.get(url, params=params, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None
    
    def configure_alerts(self, config: Dict) -> Dict:
        """Configure alert settings"""
        return self._make_request('POST', 'api/alerts/configure', json=config) or {}

    def get_alert_settings(self) -> Dict:
        """Get current alert settings"""
        return self._make_request('GET', 'api/alerts/settings') or {}

    def trigger_alert(self, payload: Dict) -> Dict:
        """Manually trigger an alert email"""
        return self._make_request('POST', 'api/alerts/trigger', json=payload) or {}
    
    def create_test_suite(self, name: str, test_ids: List[str]) -> Dict:
        """Create a custom test suite"""
        return self._make_request('POST', 'api/test-suites/create', json={'name': name, 'test_ids': test_ids}) or {}
    
    def get_test_suites(self) -> List[Dict]:
        """Get all test suites"""
        return self._make_request('GET', 'api/test-suites') or []
    
    def run_test_suite(self, suite_id: str) -> Dict:
        """Run a specific test suite"""
        return self._make_request('POST', f'api/test-suites/{suite_id}/run') or {}

    # -------------------------------------------------------------------------
    # Task 12.2: New endpoints for baselines, checkpoint, drift, attention, SHAP
    # -------------------------------------------------------------------------

    def get_chronic_failures(self, min_cycles: int = 3) -> List[Dict]:
        """Get tests that repeatedly fail across multiple cycles with root cause"""
        params = {'min_cycles': min_cycles}
        return self._make_request('GET', 'api/ai/chronic-failures', params=params) or []

    def get_prioritized_tests(self, build_cycle: str = None, drift_phase: str = "Backend,Frontend") -> Dict:
        """Get prioritized test list, critical failures, and skipped tests"""
        params = {}
        if build_cycle:
            params['build_cycle'] = build_cycle
        if drift_phase:
            params['drift_phase'] = drift_phase
        return self._make_request('GET', 'api/dashboard/prioritized-tests', params=params) or {}

    def get_baselines_apfd(self) -> Dict:
        """Get per-cycle APFD for all baselines and Semanti-Q"""
        return self._make_request('GET', 'api/baselines/apfd') or {}

    def get_checkpoint_status(self) -> Dict:
        """Get model checkpoint metadata"""
        return self._make_request('GET', 'api/model/checkpoint-status') or {}

    def get_drift_history(self) -> List[Dict]:
        """Get drift event log"""
        return self._make_request('GET', 'api/model/drift-history') or []

    def get_attention_weights(self) -> Dict:
        """Get last captured cross-attention gate values"""
        return self._make_request('GET', 'api/ai/attention-weights') or {}

    def get_shap_summary(self) -> Dict:
        """Get SHAP top-5 feature importance for most recent cycle"""
        return self._make_request('GET', 'api/ai/shap-summary') or {}

    def get_training_loss(self) -> Dict:
        """Get per-epoch training loss history"""
        return self._make_request('GET', 'api/model/training-loss') or {}

    def get_simulation_progress(self) -> Dict:
        """Get real-time simulation progress"""
        return self._make_request('GET', 'api/simulation/progress') or {}

    def get_ablation_results(self) -> List[Dict]:
        """Get extended ablation study results"""
        return self._make_request('GET', 'api/research/ablation') or []

    def get_sensitivity_results(self) -> List[Dict]:
        """Get reward weight sensitivity analysis"""
        return self._make_request('GET', 'api/research/sensitivity') or []

    def get_dataset_stats(self) -> Dict:
        """Get synthetic dataset realism statistics"""
        return self._make_request('GET', 'api/research/dataset-stats') or {}

    def get_napfd_history(self) -> Dict:
        """Get per-cycle NAPFD for all methods"""
        return self._make_request('GET', 'api/research/napfd') or {}

    def get_ttff_history(self) -> Dict:
        """Get per-cycle TTFF for all methods"""
        return self._make_request('GET', 'api/research/ttff') or {}