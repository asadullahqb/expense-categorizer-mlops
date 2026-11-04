"""
Monitoring and Alerting Module

This module provides monitoring capabilities for the expense categorizer service.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import mlflow
from dataclasses import dataclass
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

@dataclass
class Alert:
    """Alert data structure"""
    timestamp: datetime
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    service: str
    details: Optional[Dict] = None

class ExpenseCategorizerMonitor:
    """Monitoring system for the expense categorizer"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.metrics: Dict = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
    
    def log_alert(self, level: str, message: str, service: str, details: Optional[Dict] = None):
        """Log an alert"""
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            message=message,
            service=service,
            details=details
        )
        
        self.alerts.append(alert)
        
        # Log based on level
        if level == "CRITICAL":
            self.logger.critical(f"[{service}] {message}")
        elif level == "ERROR":
            self.logger.error(f"[{service}] {message}")
        elif level == "WARNING":
            self.logger.warning(f"[{service}] {message}")
        else:
            self.logger.info(f"[{service}] {message}")
    
    def check_service_health(self) -> Dict:
        """Check the health of all services"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'services': {}
        }
        
        # Check data directories
        data_dirs = ['data/raw', 'data/processed', 'mlruns']
        for dir_path in data_dirs:
            if Path(dir_path).exists():
                health_status['services'][f'directory_{dir_path}'] = 'healthy'
            else:
                health_status['services'][f'directory_{dir_path}'] = 'unhealthy'
                self.log_alert("WARNING", f"Directory {dir_path} not found", "filesystem")
        
        # Check MLflow
        try:
            mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
            experiments = mlflow.search_experiments()
            health_status['services']['mlflow'] = 'healthy'
        except Exception as e:
            health_status['services']['mlflow'] = 'unhealthy'
            self.log_alert("ERROR", f"MLflow connection failed: {str(e)}", "mlflow")
        
        # Check for recent data processing
        processed_dir = Path("data/processed")
        if processed_dir.exists():
            recent_files = list(processed_dir.glob("*.csv"))
            if recent_files:
                latest_file = max(recent_files, key=lambda f: f.stat().st_mtime)
                file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
                
                if file_age > timedelta(days=7):
                    self.log_alert("WARNING", "No recent data processing", "data_pipeline")
                    health_status['services']['data_freshness'] = 'stale'
                else:
                    health_status['services']['data_freshness'] = 'fresh'
            else:
                health_status['services']['data_freshness'] = 'no_data'
                self.log_alert("WARNING", "No processed data files found", "data_pipeline")
        
        # Determine overall status
        unhealthy_services = [s for s in health_status['services'].values() if s != 'healthy']
        if unhealthy_services:
            health_status['overall_status'] = 'unhealthy'
            self.log_alert("ERROR", f"Unhealthy services: {unhealthy_services}", "monitoring")
        
        return health_status
    
    def check_data_quality(self) -> Dict:
        """Check data quality metrics"""
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_quality': 'good',
            'issues': []
        }
        
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            quality_report['issues'].append("No processed data directory")
            quality_report['overall_quality'] = 'poor'
            return quality_report
        
        # Check recent files
        recent_files = list(processed_dir.glob("*.csv"))
        if not recent_files:
            quality_report['issues'].append("No processed data files")
            quality_report['overall_quality'] = 'poor'
            return quality_report
        
        # Analyze latest file
        latest_file = max(recent_files, key=lambda f: f.stat().st_mtime)
        try:
            df = pd.read_csv(latest_file)
            
            # Check for missing values
            missing_data = df.isnull().sum()
            if missing_data.any():
                quality_report['issues'].append(f"Missing data in columns: {missing_data[missing_data > 0].to_dict()}")
            
            # Check for duplicate transactions
            if 'description' in df.columns and 'amount' in df.columns:
                duplicates = df.duplicated(subset=['description', 'amount']).sum()
                if duplicates > 0:
                    quality_report['issues'].append(f"Found {duplicates} duplicate transactions")
            
            # Check amount distribution
            if 'amount' in df.columns:
                amount_stats = df['amount'].describe()
                if amount_stats['std'] == 0:
                    quality_report['issues'].append("All amounts are identical (suspicious)")
                
                # Check for negative amounts (might be legitimate)
                negative_count = (df['amount'] < 0).sum()
                if negative_count > len(df) * 0.5:
                    quality_report['issues'].append(f"High percentage of negative amounts: {negative_count/len(df)*100:.1f}%")
            
            # Check category distribution
            if 'category' in df.columns:
                category_counts = df['category'].value_counts()
                if len(category_counts) == 1:
                    quality_report['issues'].append("All transactions have the same category")
                
                # Check for 'Other' category dominance
                if 'Other' in category_counts and category_counts['Other'] > len(df) * 0.8:
                    quality_report['issues'].append("High percentage of 'Other' category transactions")
            
        except Exception as e:
            quality_report['issues'].append(f"Error analyzing data: {str(e)}")
            quality_report['overall_quality'] = 'poor'
        
        # Determine overall quality
        if len(quality_report['issues']) > 3:
            quality_report['overall_quality'] = 'poor'
        elif len(quality_report['issues']) > 1:
            quality_report['overall_quality'] = 'fair'
        
        return quality_report
    
    def check_model_performance(self) -> Dict:
        """Check ML model performance metrics"""
        performance_report = {
            'timestamp': datetime.now().isoformat(),
            'model_status': 'unknown',
            'metrics': {}
        }
        
        try:
            mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
            
            # Get latest experiment
            experiments = mlflow.search_experiments()
            if not experiments:
                performance_report['model_status'] = 'no_experiments'
                return performance_report
            
            # Get latest run
            runs = mlflow.search_runs(experiment_ids=[experiments[0].experiment_id])
            if runs.empty:
                performance_report['model_status'] = 'no_runs'
                return performance_report
            
            latest_run = runs.iloc[0]
            
            # Check for accuracy metric
            if 'metrics.accuracy' in latest_run:
                accuracy = latest_run['metrics.accuracy']
                performance_report['metrics']['accuracy'] = accuracy
                
                if accuracy < 0.7:
                    self.log_alert("WARNING", f"Low model accuracy: {accuracy:.3f}", "ml_model")
                    performance_report['model_status'] = 'poor'
                elif accuracy < 0.85:
                    performance_report['model_status'] = 'fair'
                else:
                    performance_report['model_status'] = 'good'
            else:
                performance_report['model_status'] = 'no_metrics'
                self.log_alert("WARNING", "No accuracy metrics found in latest run", "ml_model")
            
        except Exception as e:
            performance_report['model_status'] = 'error'
            self.log_alert("ERROR", f"Error checking model performance: {str(e)}", "ml_model")
        
        return performance_report
    
    def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        resource_report = {
            'timestamp': datetime.now().isoformat(),
            'disk_usage': {},
            'file_counts': {}
        }
        
        try:
            # Check disk usage for data directories
            data_dirs = ['data', 'mlruns', 'models']
            for dir_path in data_dirs:
                if Path(dir_path).exists():
                    # Get directory size
                    total_size = sum(f.stat().st_size for f in Path(dir_path).rglob('*') if f.is_file())
                    resource_report['disk_usage'][dir_path] = {
                        'size_mb': total_size / (1024 * 1024),
                        'file_count': len(list(Path(dir_path).rglob('*')))
                    }
                    
                    # Alert if directory is getting large
                    if total_size > 100 * 1024 * 1024:  # 100MB
                        self.log_alert("WARNING", f"Large directory size: {dir_path} ({total_size/(1024*1024):.1f}MB)", "disk_usage")
            
        except Exception as e:
            self.log_alert("ERROR", f"Error checking system resources: {str(e)}", "system")
        
        return resource_report
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'health': self.check_service_health(),
            'data_quality': self.check_data_quality(),
            'model_performance': self.check_model_performance(),
            'system_resources': self.check_system_resources(),
            'recent_alerts': [alert.__dict__ for alert in self.alerts[-10:]]  # Last 10 alerts
        }
        
        # Save report
        report_path = f"logs/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring report saved: {report_path}")
        return report
    
    def send_alert_notification(self, alert: Alert, email_config: Optional[Dict] = None):
        """Send alert notification via email"""
        if not email_config:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']
            msg['Subject'] = f"[{alert.level}] Expense Categorizer Alert"
            
            body = f"""
            Alert Details:
            - Time: {alert.timestamp}
            - Level: {alert.level}
            - Service: {alert.service}
            - Message: {alert.message}
            """
            
            if alert.details:
                body += f"\nDetails: {json.dumps(alert.details, indent=2)}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Alert notification sent for {alert.service}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert notification: {str(e)}")

def main():
    """Main function for running monitoring"""
    monitor = ExpenseCategorizerMonitor()
    
    print("🔍 Running expense categorizer monitoring...")
    
    # Generate comprehensive report
    report = monitor.generate_monitoring_report()
    
    # Print summary
    print(f"\n📊 Monitoring Summary:")
    print(f"Overall Health: {report['health']['overall_status']}")
    print(f"Data Quality: {report['data_quality']['overall_quality']}")
    print(f"Model Status: {report['model_performance']['model_status']}")
    print(f"Recent Alerts: {len(report['recent_alerts'])}")
    
    # Print any issues
    if report['data_quality']['issues']:
        print(f"\n⚠️ Data Quality Issues:")
        for issue in report['data_quality']['issues']:
            print(f"  - {issue}")
    
    if report['recent_alerts']:
        print(f"\n🚨 Recent Alerts:")
        for alert in report['recent_alerts'][-5:]:  # Last 5 alerts
            print(f"  [{alert['level']}] {alert['message']}")

if __name__ == "__main__":
    main()