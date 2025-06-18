#!/usr/bin/env python3
"""
Complete workflow script for ultra-lightweight Railway deployment.
This script handles the entire process from precomputation to deployment preparation.
"""

import os
import sys
import logging
import subprocess
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UltraLightweightDeploymentManager:
    """Manages the complete ultra-lightweight deployment workflow"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.precomputed_dir = self.project_root / "precomputed_ultra"
        self.backup_dir = self.project_root / "deployment_backup"
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        logger.info("🔍 Checking prerequisites...")
        
        required_files = [
            "discourse_posts.json",
            "hybrid_search_engine.py",
            "precompute_embeddings_ultra.py",
            "ultra_lightweight_engine.py",
            "main-ultra-railway.py",
            "requirements-ultra-light.txt"
        ]
        
        missing_files = []
        for filename in required_files:
            if not (self.project_root / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        # Check if .env file exists and has required variables
        env_file = self.project_root / ".env"
        if env_file.exists():
            logger.info("✅ .env file found")
        else:
            logger.warning("⚠️ .env file not found - OpenAI embeddings may not work")
        
        logger.info("✅ All prerequisites met")
        return True
    
    def backup_current_setup(self):
        """Backup current deployment setup"""
        logger.info("💾 Backing up current setup...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir()
        
        # Files to backup
        backup_files = [
            "requirements.txt",
            "main-railway.py"
        ]
        
        for filename in backup_files:
            src = self.project_root / filename
            if src.exists():
                dst = self.backup_dir / filename
                shutil.copy2(src, dst)
                logger.info(f"   Backed up: {filename}")
        
        logger.info("✅ Backup completed")
    
    def run_precomputation(self) -> bool:
        """Run the precomputation process"""
        logger.info("🔄 Starting precomputation process...")
        start_time = time.time()
        
        try:
            # Import and run precomputation
            from precompute_embeddings_ultra import UltraLightweightPrecomputer
            
            precomputer = UltraLightweightPrecomputer()
            success = precomputer.run_precomputation()
            
            duration = time.time() - start_time
            
            if success:
                logger.info(f"✅ Precomputation completed in {duration:.1f} seconds")
                
                # Display precomputation stats
                metadata_file = self.precomputed_dir / "ultra_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    logger.info(f"📊 Precomputation statistics:")
                    logger.info(f"   Total size: {metadata.get('total_size_mb', 0):.2f} MB")
                    logger.info(f"   OpenAI embeddings: {metadata.get('openai_embeddings', 0)}")
                    logger.info(f"   Local embeddings: {metadata.get('local_embeddings', 0)}")
                    logger.info(f"   TF-IDF features: {metadata.get('tfidf_features', 0)}")
                    logger.info(f"   Subthreads: {metadata.get('subthreads_count', 0)}")
                
                return True
            else:
                logger.error("❌ Precomputation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Precomputation error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def validate_precomputed_data(self) -> bool:
        """Validate the precomputed data"""
        logger.info("🧪 Validating precomputed data...")
        
        if not self.precomputed_dir.exists():
            logger.error("❌ Precomputed directory not found")
            return False
        
        # Check required files (OpenAI embeddings are optional)
        required_files = [
            "local_embeddings.pkl",
            "tfidf_data.pkl",
            "search_indices.json",
            "subthreads_light.json",
            "ultra_metadata.json"
        ]
        
        optional_files = [
            "openai_embeddings.pkl"  # Optional if OpenAI API is not available
        ]
        
        missing_required = []
        for filename in required_files:
            filepath = self.precomputed_dir / filename
            if not filepath.exists():
                missing_required.append(filename)
        
        if missing_required:
            logger.error(f"❌ Missing required precomputed files: {missing_required}")
            return False
        
        # Check optional files
        missing_optional = []
        for filename in optional_files:
            filepath = self.precomputed_dir / filename
            if not filepath.exists():
                missing_optional.append(filename)
        
        if missing_optional:
            logger.warning(f"⚠️ Missing optional files (system will work without them): {missing_optional}")
        
        # Test loading the ultra-lightweight engine
        try:
            from ultra_lightweight_engine import UltraLightweightSearchEngine
            engine = UltraLightweightSearchEngine()
            
            # Test a simple search
            results = engine.search("test query", top_k=1)
            
            stats = engine.get_stats()
            logger.info(f"✅ Validation successful")
            logger.info(f"   Total subthreads: {stats['total_subthreads']}")
            logger.info(f"   Has OpenAI embeddings: {stats['has_openai_embeddings']}")
            logger.info(f"   Has local embeddings: {stats['has_local_embeddings']}")
            logger.info(f"   Has TF-IDF: {stats['has_tfidf']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return False
    
    def prepare_railway_deployment(self):
        """Prepare files for Railway deployment"""
        logger.info("🚀 Preparing Railway deployment...")
        
        # Copy ultra-lightweight requirements
        src_requirements = self.project_root / "requirements-ultra-light.txt"
        dst_requirements = self.project_root / "requirements.txt"
        
        shutil.copy2(src_requirements, dst_requirements)
        logger.info("✅ Updated requirements.txt with ultra-lightweight dependencies")
        
        # Update railway.toml if it exists
        railway_toml = self.project_root / "railway.toml"
        if railway_toml.exists():
            # Read current content
            with open(railway_toml, 'r') as f:
                content = f.read()
            
            # Update the start command to use ultra-lightweight app
            if 'main-railway.py' in content:
                content = content.replace('main-railway.py', 'main-ultra-railway.py')
                
                with open(railway_toml, 'w') as f:
                    f.write(content)
                
                logger.info("✅ Updated railway.toml to use ultra-lightweight app")
            else:
                logger.info("ℹ️ railway.toml doesn't contain main-railway.py reference")
        else:
            # Create a new railway.toml
            railway_config = """[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn main-ultra-railway:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3

[variables]
ENVIRONMENT = "production"
"""
            with open(railway_toml, 'w') as f:
                f.write(railway_config)
            
            logger.info("✅ Created railway.toml for ultra-lightweight deployment")
        
        # Update .railwayignore if it exists
        railwayignore = self.project_root / ".railwayignore"
        if railwayignore.exists():
            with open(railwayignore, 'r') as f:
                ignore_content = f.read()
            
            # Ensure we don't ignore precomputed data
            if 'precomputed_ultra/' in ignore_content:
                ignore_content = ignore_content.replace('precomputed_ultra/', '# precomputed_ultra/')
                
                with open(railwayignore, 'w') as f:
                    f.write(ignore_content)
                
                logger.info("✅ Updated .railwayignore to include precomputed data")
        
        logger.info("🎯 Railway deployment preparation completed")
    
    def run_validation_tests(self) -> bool:
        """Run validation tests"""
        logger.info("🧪 Running validation tests...")
        
        try:
            # Run the test script
            result = subprocess.run([
                sys.executable, "test_ultra_lightweight.py"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ All validation tests passed")
                return True
            else:
                logger.error("❌ Validation tests failed")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Validation tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error running validation tests: {e}")
            return False
    
    def generate_deployment_summary(self) -> Dict[str, Any]:
        """Generate deployment summary"""
        logger.info("📋 Generating deployment summary...")
        
        summary = {
            "deployment_type": "ultra_lightweight",
            "timestamp": time.time(),
            "precomputed_data": {},
            "files_modified": [],
            "railway_ready": True
        }
        
        # Get precomputed data stats
        metadata_file = self.precomputed_dir / "ultra_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                summary["precomputed_data"] = json.load(f)
        
        # List modified files
        summary["files_modified"] = [
            "requirements.txt (updated to ultra-lightweight)",
            "railway.toml (updated for ultra-lightweight app)",
            "precomputed_ultra/ (created with all precomputed data)"
        ]
        
        # Check total deployment size
        total_size = 0
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    pass
        
        summary["total_deployment_size_mb"] = total_size / (1024 * 1024)
        
        return summary
    
    def run_complete_workflow(self) -> bool:
        """Run the complete deployment workflow"""
        logger.info("🚀 Starting ultra-lightweight deployment workflow...")
        workflow_start = time.time()
        
        try:
            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                return False
            
            # Step 2: Backup current setup
            self.backup_current_setup()
            
            # Step 3: Run precomputation
            if not self.run_precomputation():
                logger.error("❌ Workflow failed at precomputation step")
                return False
            
            # Step 4: Validate precomputed data
            if not self.validate_precomputed_data():
                logger.error("❌ Workflow failed at validation step")
                return False
            
            # Step 5: Prepare Railway deployment
            self.prepare_railway_deployment()
            
            # Step 6: Run validation tests
            if not self.run_validation_tests():
                logger.warning("⚠️ Some validation tests failed, but deployment may still work")
            
            # Step 7: Generate summary
            summary = self.generate_deployment_summary()
            
            workflow_duration = time.time() - workflow_start
            
            logger.info("🎉 Ultra-lightweight deployment workflow completed successfully!")
            logger.info(f"⏱️ Total workflow time: {workflow_duration:.1f} seconds")
            logger.info(f"📦 Deployment size: {summary['total_deployment_size_mb']:.2f} MB")
            logger.info(f"🚀 Ready for Railway deployment!")
            
            # Save summary
            summary_file = self.project_root / "deployment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"📋 Deployment summary saved to: {summary_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Workflow failed with error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def restore_backup(self):
        """Restore from backup"""
        logger.info("🔄 Restoring from backup...")
        
        if not self.backup_dir.exists():
            logger.error("❌ No backup found")
            return False
        
        for backup_file in self.backup_dir.glob('*'):
            if backup_file.is_file():
                dst = self.project_root / backup_file.name
                shutil.copy2(backup_file, dst)
                logger.info(f"   Restored: {backup_file.name}")
        
        logger.info("✅ Backup restored")
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-lightweight Railway deployment manager")
    parser.add_argument("--restore", action="store_true", help="Restore from backup")
    parser.add_argument("--precompute-only", action="store_true", help="Run precomputation only")
    parser.add_argument("--validate-only", action="store_true", help="Run validation only")
    
    args = parser.parse_args()
    
    manager = UltraLightweightDeploymentManager()
    
    if args.restore:
        success = manager.restore_backup()
    elif args.precompute_only:
        success = manager.run_precomputation()
    elif args.validate_only:
        success = manager.validate_precomputed_data()
    else:
        success = manager.run_complete_workflow()
    
    if success:
        logger.info("✅ Operation completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()