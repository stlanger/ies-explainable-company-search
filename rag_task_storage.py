import sqlite3


class RAG_Task_Storage:
    def __init__(self, get_db_func):
        self.get_db = get_db_func

    def initialize_database(self):
        """Creates the SQLite database and tables if they do not exist."""
        conn = self.get_db()
        cursor = conn.cursor()

        # Create tasks table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            query TEXT NOT NULL,            
            created DATETIME DEFAULT CURRENT_TIMESTAMP,
            processing_start DATETIME,
            processing_end DATETIME
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks_candidate_sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            sentence TEXT NOT NULL,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id) ON DELETE CASCADE
        )
        """)
        
  
        # Create results table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            score INTEGER NOT NULL,
            company_id TEXT NOT NULL,
            name TEXT NOT NULL,
            website TEXT NOT NULL,
            explanation TEXT NOT NULL,
            context TEXT NOT NULL,
            base_vector_enhanced_score REAL NOT NULL,
            FOREIGN KEY (task_id) REFERENCES tasks (task_id) ON DELETE CASCADE
        )
        """)

        # ### SIGIR EVALUATION TABLES
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS session (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            type_id INTEGER NOT NULL,
            created DATETIME NOT NULL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            task_id TEXT NOT NULL,            
            finished DATETIME,
            FOREIGN KEY (session_id) REFERENCES session (session_id) ON DELETE CASCADE,
            FOREIGN KEY (task_id) REFERENCES session (task_id) ON DELETE CASCADE
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_task_id INTEGER NOT NULL,
            result_id INTEGER NOT NULL,            
            rating INTEGER NOT NULL, 
            website_clicked BOOLEAN NOT NULL,
            first_interaction_time DATETIME,            
            FOREIGN KEY (session_task_id) REFERENCES session (id) ON DELETE CASCADE,
            FOREIGN KEY (result_id) REFERENCES results (id) ON DELETE CASCADE        )
        """)

        conn.commit()

  
    def insert_task(self, task_id, query, processing_start=None, processing_end=None):
        conn = self.get_db()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO tasks (task_id, query, processing_start, processing_end)
        VALUES (?, ?, ?, ?)
        """, (task_id, query, processing_start, processing_end)) 

        conn.commit()

    def insert_task_candidate_sentences(self, task_id, sentences):
        conn = self.get_db()
        cursor = conn.cursor()

        for sentence in sentences:
            cursor.execute("""
            INSERT INTO tasks_candidate_sentences(task_id, sentence)
            VALUES (?, ?)
            """, (task_id, sentence)) 

        conn.commit()

    def insert_result(self, task_id, score, company_id, name, website, explanation, context):
        conn = self.get_db()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO results (task_id, score, name, website, explanation, context)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (task_id, score, company_id, name, website, explanation, context))

        conn.commit()

    def insert_results(self, task_id, results):
        conn = self.get_db()
        cursor = conn.cursor()

        for result in results:
            cursor.execute("""
            INSERT INTO results (task_id, score, base_vector_enhanced_score, company_id, name, website, explanation, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (task_id, result['score'], result['base_vector_enhanced_score'], result['company_id'], result['name'], result['website'], result['explanation'], result['context']))

        conn.commit()

    def set_start_time(self, task_id):
        conn = self.get_db()
        cursor = conn.cursor()

        cursor.execute("""
        UPDATE tasks
        SET processing_start = CURRENT_TIMESTAMP
        WHERE task_id = ?
        """, (task_id,))
        conn.commit()

    def set_end_time(self, task_id):
        conn = self.get_db()
        cursor = conn.cursor()

        cursor.execute("""
        UPDATE tasks
        SET processing_end = CURRENT_TIMESTAMP
        WHERE task_id = ?
        """, (task_id,))
        conn.commit()

    def list_tasks(self, blacklist=None):
        if not blacklist:
            blacklist = []
    
        conn = self.get_db()
        cursor = conn.cursor()
    
        if blacklist:
            placeholders = ",".join("?" for _ in blacklist)
            sql = f"""
            SELECT id, task_id, query, created, processing_start, processing_end
            FROM tasks
            WHERE task_id NOT IN ({placeholders})
            ORDER BY created DESC
            """
            cursor.execute(sql, tuple(blacklist))
        else:
            sql = """
            SELECT id, task_id, query, created, processing_start, processing_end
            FROM tasks
            ORDER BY created DESC
            """
            cursor.execute(sql)
    
        rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "task_id": row[1],
                "query": row[2],
                "created": row[3],
                "processing_start": row[4],
                "processing_end": row[5],
            }
            for row in rows
        ]

    def task_details(self, task_id):
        conn = self.get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, task_id, query, created, processing_start, processing_end
            FROM tasks
            WHERE task_id = ?
            ORDER BY created DESC
        """, (task_id,))

        row = cursor.fetchone()
        return {
            "id": row[0],
            "task_id": row[1],
            "query": row[2],
            "created": row[3],
            "processing_start": row[4],
            "processing_end": row[5],
        }
        

    def list_unfinished_tasks(self):
        conn = self.get_db()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT id, task_id, query, created, processing_start, processing_end
        FROM tasks
        WHERE processing_end IS NULL
        ORDER BY created DESC
        """)

        rows = cursor.fetchall()
        print(f"{len(rows)} unfinished tasks.")
        return [
            {
                "id": row[0],
                "task_id": row[1],
                "query": row[2],
                "created": row[3],
                "processing_start": row[4],
                "processing_end": row[5],
            }
            for row in rows
        ]
        

    def list_results_for_task(self, task_id):
        conn = self.get_db()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT id, task_id, score, company_id, name, website, explanation
        FROM results
        WHERE task_id = ?
        ORDER BY score DESC
        """, (task_id,))
        
        rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "task_id": row[1],
                "score": row[2],
                "company_id": row[3],
                "name": row[4],
                "website": row[5],
                "explanation": row[6],
            }
            for row in rows if isinstance(row[2], int)
        ]


    def insert_session(self, session_id, type_id):
        conn = self.get_db()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO session (session_id, type_id, created)
        VALUES (?, ?, DateTime('now'))
        """, (session_id, type_id)) 

        conn.commit()


    def insert_session_tasks(self, session_id, task_ids):
        conn = self.get_db()
        cursor = conn.cursor()

        for task_id in task_ids:
            cursor.execute("""
            INSERT INTO session_tasks (session_id, task_id)
            VALUES (?, ?)
            """, (session_id, task_id)) 

        conn.commit()

    def insert_user_feedback(self, session_id, task_id, feedback):
        conn = self.get_db()
        cursor = conn.cursor()

        cursor.execute("""UPDATE session_tasks SET finished=CURRENT_TIMESTAMP WHERE session_id=? AND task_id=?""", (session_id, task_id))

        cursor.execute("""SELECT id FROM session_tasks WHERE session_id=? AND task_id=?;""", (session_id, task_id))
        rows = cursor.fetchall()
        session_task_id = rows[0][0]
        
        for item in feedback:
            print("insert_user_feedback:", item)
            cursor.execute("""
            INSERT INTO feedback(session_task_id, result_id, rating, website_clicked, first_interaction_time)
            VALUES (?, ?, ?, ?, ?)
            """, (session_task_id, item["result_id"], item["rating"], item["website_clicked"], item["first_interaction_time"])) 
        
        conn.commit()