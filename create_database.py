import pandas as pd
import sqlite3
import os
import re
from datetime import datetime
import logging
from pathlib import Path
import numpy as np

class FuturesOptionsProcessor:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.futures_path = self.base_path / "futures_data_project" / "future_data"
        self.options_path = self.base_path / "futures_data_project" / "option_data"
        self.output_path = self.base_path / "futures_data_project" / "output"
        self.db_path = self.output_path / "database" / "trading_data.db"
        
        # Create output directories
        os.makedirs(self.output_path / "database", exist_ok=True)
        os.makedirs(self.output_path / "processed_data", exist_ok=True)
        os.makedirs(self.output_path / "logs", exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=self.output_path / "logs" / "processing.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'  # Overwrite log file each run
        )
        
        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        
    def setup_database(self):
        """Create database and tables"""
        logging.info("Setting up database...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create futures table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS futures_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_recorded DATE,
            tradable TEXT,
            mode TEXT,
            tradingsymbol TEXT,
            token INTEGER,
            ltp REAL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            change REAL,
            et TEXT,
            ltq INTEGER,
            atp REAL,
            volume INTEGER,
            tbq INTEGER,
            tsq INTEGER,
            ltt TEXT,
            oi INTEGER,
            oi_high INTEGER,
            oi_low INTEGER,
            B1Q INTEGER, B1P REAL, B1O INTEGER,
            B2Q INTEGER, B2P REAL, B2O INTEGER,
            B3Q INTEGER, B3P REAL, B3O INTEGER,
            B4Q INTEGER, B4P REAL, B4O INTEGER,
            B5Q INTEGER, B5P REAL, B5O INTEGER,
            A1Q INTEGER, A1P REAL, A1O INTEGER,
            A2Q INTEGER, A2P REAL, A2O INTEGER,
            A3Q INTEGER, A3P REAL, A3O INTEGER,
            A4Q INTEGER, A4P REAL, A4O INTEGER,
            A5Q INTEGER, A5P REAL, A5O INTEGER
        )
        ''')
        
        # Create options table (identical structure to futures)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS options_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_recorded DATE,
            tradable TEXT,
            mode TEXT,
            tradingsymbol TEXT,
            token INTEGER,
            ltp REAL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            change REAL,
            et TEXT,
            ltq INTEGER,
            atp REAL,
            volume INTEGER,
            tbq INTEGER,
            tsq INTEGER,
            ltt TEXT,
            oi INTEGER,
            oi_high INTEGER,
            oi_low INTEGER,
            B1Q INTEGER, B1P REAL, B1O INTEGER,
            B2Q INTEGER, B2P REAL, B2O INTEGER,
            B3Q INTEGER, B3P REAL, B3O INTEGER,
            B4Q INTEGER, B4P REAL, B4O INTEGER,
            B5Q INTEGER, B5P REAL, B5O INTEGER,
            A1Q INTEGER, A1P REAL, A1O INTEGER,
            A2Q INTEGER, A2P REAL, A2O INTEGER,
            A3Q INTEGER, A3P REAL, A3O INTEGER,
            A4Q INTEGER, A4P REAL, A4O INTEGER,
            A5Q INTEGER, A5P REAL, A5O INTEGER,
            option_type TEXT,
            strike_price REAL,
            expiry_date TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        logging.info("Database setup completed")
        
    def process_csv_file(self, file_path, table_name):
        """Process a single CSV file"""
        try:
            logging.info(f"Processing {file_path}")
            
            # Extract date from filename
            filename = os.path.basename(file_path)
            date_str = filename.split('_')[1].split('.')[0]  # Extract YYYY-MM-DD
            
            # Try different separators and read methods
            df = None
            separators = [' ', '\t', ',']
            
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, sep=sep)
                    # Check if we have the expected columns
                    if 'tradingsymbol' in df.columns and len(df.columns) > 30:
                        logging.info(f"Successfully read file with separator '{sep}'")
                        break
                except:
                    continue
            
            if df is None or df.empty:
                logging.error(f"Could not read file {file_path}")
                return None
            
            # Add date column
            df['date_recorded'] = date_str
            
            # Clean data
            df = self.clean_data(df, table_name)
            
            return df
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return None
    
    def clean_data(self, df, table_name):
        """Clean and validate data"""
        try:
            # Handle boolean columns
            if 'tradable' in df.columns:
                df['tradable'] = df['tradable'].astype(str)
            
            # Handle numeric columns
            numeric_cols = ['token', 'ltp', 'open', 'high', 'low', 'close', 'change', 
                           'ltq', 'atp', 'volume', 'tbq', 'tsq', 'oi', 'oi_high', 'oi_low']
            
            # Add bid/ask columns
            for level in range(1, 6):
                numeric_cols.extend([f'B{level}Q', f'B{level}P', f'B{level}O'])
                numeric_cols.extend([f'A{level}Q', f'A{level}P', f'A{level}O'])
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # For options data, extract additional info
            if table_name == 'options_data':
                df = self.extract_options_info(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Error cleaning data: {e}")
            return df
    
    def extract_options_info(self, df):
        """Extract option type, strike price, and expiry from trading symbol"""
        def parse_option_symbol(symbol):
            try:
                if pd.isna(symbol):
                    return None, None, None
                    
                symbol = str(symbol).upper()
                
                # Determine option type
                if 'CE' in symbol:
                    option_type = 'CE'
                    base_symbol = symbol.replace('CE', '')
                elif 'PE' in symbol:
                    option_type = 'PE' 
                    base_symbol = symbol.replace('PE', '')
                else:
                    return None, None, None
                
                # Extract strike price (last numeric part before CE/PE)
                strike_match = re.search(r'(\d+)(?=CE|PE)', symbol)
                strike_price = float(strike_match.group(1)) if strike_match else None
                
                # Extract expiry info
                expiry_match = re.search(r'(\d{2}[A-Z]{3})', symbol)
                expiry_str = expiry_match.group(1) if expiry_match else None
                
                return option_type, strike_price, expiry_str
                
            except Exception as e:
                logging.warning(f"Could not parse option symbol {symbol}: {e}")
                return None, None, None
        
        # Apply to all rows
        option_info = df['tradingsymbol'].apply(parse_option_symbol)
        df['option_type'] = [x[0] for x in option_info]
        df['strike_price'] = [x[1] for x in option_info]
        df['expiry_date'] = [x[2] for x in option_info]
        
        return df
    
    def process_all_futures(self):
        """Process all futures files"""
        logging.info("Processing futures files...")
        
        if not self.futures_path.exists():
            logging.error(f"Futures path does not exist: {self.futures_path}")
            return
        
        futures_files = list(self.futures_path.glob("Futures_*.csv"))
        logging.info(f"Found {len(futures_files)} futures files")
        
        if not futures_files:
            logging.warning("No futures files found")
            return
        
        conn = sqlite3.connect(self.db_path)
        total_records = 0
        
        for file_path in futures_files:
            df = self.process_csv_file(file_path, 'futures_data')
            
            if df is not None and not df.empty:
                try:
                    df.to_sql('futures_data', conn, if_exists='append', index=False)
                    logging.info(f"Inserted {len(df)} records from {file_path}")
                    total_records += len(df)
                except Exception as e:
                    logging.error(f"Error inserting data from {file_path}: {e}")
        
        conn.close()
        logging.info(f"Total futures records processed: {total_records}")
    
    def process_all_options(self):
        """Process all options files"""
        logging.info("Processing options files...")
        
        if not self.options_path.exists():
            logging.error(f"Options path does not exist: {self.options_path}")
            return
        
        options_files = list(self.options_path.glob("Options_*.csv"))
        logging.info(f"Found {len(options_files)} options files")
        
        if not options_files:
            logging.warning("No options files found")
            return
        
        conn = sqlite3.connect(self.db_path)
        total_records = 0
        
        for file_path in options_files:
            df = self.process_csv_file(file_path, 'options_data')
            
            if df is not None and not df.empty:
                try:
                    df.to_sql('options_data', conn, if_exists='append', index=False)
                    logging.info(f"Inserted {len(df)} records from {file_path}")
                    total_records += len(df)
                except Exception as e:
                    logging.error(f"Error inserting data from {file_path}: {e}")
        
        conn.close()
        logging.info(f"Total options records processed: {total_records}")
    
    def create_indexes(self):
        """Create database indexes for better performance"""
        logging.info("Creating database indexes...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_futures_date ON futures_data(date_recorded)",
            "CREATE INDEX IF NOT EXISTS idx_futures_symbol ON futures_data(tradingsymbol)",
            "CREATE INDEX IF NOT EXISTS idx_futures_token ON futures_data(token)",
            "CREATE INDEX IF NOT EXISTS idx_options_date ON options_data(date_recorded)",
            "CREATE INDEX IF NOT EXISTS idx_options_symbol ON options_data(tradingsymbol)",
            "CREATE INDEX IF NOT EXISTS idx_options_type ON options_data(option_type)",
            "CREATE INDEX IF NOT EXISTS idx_options_strike ON options_data(strike_price)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                logging.info(f"Created index: {index_sql.split()[-1]}")
            except Exception as e:
                logging.error(f"Error creating index: {e}")
        
        conn.commit()
        conn.close()
        logging.info("Database indexes created")

def validate_database(db_path):
    """Validate the created database"""
    print("\n" + "="*50)
    print("DATABASE VALIDATION REPORT")
    print("="*50)
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Check record counts
        futures_count = pd.read_sql("SELECT COUNT(*) as count FROM futures_data", conn).iloc[0]['count']
        options_count = pd.read_sql("SELECT COUNT(*) as count FROM options_data", conn).iloc[0]['count']
        
        print(f"Total futures records: {futures_count:,}")
        print(f"Total options records: {options_count:,}")
        
        if futures_count > 0:
            # Check date range for futures
            futures_dates = pd.read_sql("SELECT MIN(date_recorded) as min_date, MAX(date_recorded) as max_date FROM futures_data", conn)
            print(f"Futures date range: {futures_dates.iloc[0]['min_date']} to {futures_dates.iloc[0]['max_date']}")
            
            # Check unique futures symbols
            futures_symbols = pd.read_sql("SELECT COUNT(DISTINCT tradingsymbol) as unique_symbols FROM futures_data", conn).iloc[0]['unique_symbols']
            print(f"Unique futures symbols: {futures_symbols}")
        
        if options_count > 0:
            # Check date range for options
            options_dates = pd.read_sql("SELECT MIN(date_recorded) as min_date, MAX(date_recorded) as max_date FROM options_data", conn)
            print(f"Options date range: {options_dates.iloc[0]['min_date']} to {options_dates.iloc[0]['max_date']}")
            
            # Check options breakdown
            options_breakdown = pd.read_sql("""
            SELECT 
                option_type,
                COUNT(*) as count,
                COUNT(DISTINCT strike_price) as unique_strikes
            FROM options_data 
            WHERE option_type IS NOT NULL
            GROUP BY option_type
            """, conn)
            
            if not options_breakdown.empty:
                print("\nOptions breakdown:")
                for _, row in options_breakdown.iterrows():
                    print(f"  {row['option_type']}: {row['count']:,} records, {row['unique_strikes']} unique strikes")
        
        conn.close()
        print("="*50)
        
    except Exception as e:
        print(f"Error validating database: {e}")

def main():
    """Main execution function"""
    print("Starting Futures & Options Database Creation...")
    print("="*60)
    
    # Set base path
    base_path = r"C:\Users\hp\Documents\python for trader\Complete Futures Data Processing Project"
    
    try:
        # Initialize processor
        processor = FuturesOptionsProcessor(base_path)
        
        # Setup database
        processor.setup_database()
        
        # Process all data
        processor.process_all_futures()
        processor.process_all_options()
        
        # Create indexes
        processor.create_indexes()
        
        # Validate results
        validate_database(processor.db_path)
        
        print("\n✅ Database creation completed successfully!")
        print(f"Database location: {processor.db_path}")
        print(f"Log file: {processor.output_path / 'logs' / 'processing.log'}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        logging.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()
