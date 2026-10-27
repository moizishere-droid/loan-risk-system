"""Account and record-isolation tests with a disposable database and stubbed ML."""
import importlib
import os
import sys
import types
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

os.environ['DATABASE_URL'] = 'sqlite://'

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from api import auth
from database import database as db


PAYLOAD = dict(cnic='42101-1234567-1', loan_amnt=10000, loan_grade='A', loan_intent='PERSONAL',
               loan_percent_income=0.2, person_income=50000, person_age=30, person_emp_length=5,
               person_home_ownership='RENT', cb_person_default_on_file='N', cb_person_cred_hist_length=5)


def prediction(inputs, existing=False):
    return {'workflow': 'existing_loan' if existing else 'new_applicant', 'input_data': inputs,
            'prediction_data': {
                'classification': {'decision': 'APPROVED', 'default_probability': 0.084,
                                   'threshold': 0.5, 'reasons_for_default': [], 'reasons_against_default': []},
                'regression': None if existing else {'predicted_interest_rate': 11.2,
                                                     'reasons_high_rate': [], 'reasons_low_rate': []},
                'clustering': {'cluster_id': 0, 'cluster_label': 'High Value Borrower',
                               'probabilities': {'High Value Borrower': 0.78, 'Standard Borrower': 0.22}},
                'plain_reasons': {'decision_reasons': ['No previous default.'], 'rate_reasons': []}}}


class AccountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = types.ModuleType('api.model_loader')
        loader.is_loaded = lambda: True
        loader.get_model_status = lambda: {'database': True, 'cls_onnx': True}
        pipeline = types.ModuleType('api.pipeline')
        pipeline.run_new_applicant = prediction
        pipeline.run_existing_loan = lambda inputs: prediction(inputs, True)
        with patch.dict(sys.modules, {'api.model_loader': loader, 'api.pipeline': pipeline}):
            cls.routes = importlib.import_module('api.routes')
        cls.engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
        event.listen(cls.engine, 'connect', lambda conn, _: conn.execute('PRAGMA foreign_keys=ON'))
        cls.sessions = sessionmaker(bind=cls.engine)
        cls.app = FastAPI()
        cls.app.include_router(auth.router)
        cls.app.include_router(cls.routes.router)

        def dependency():
            with cls.sessions() as session:
                yield session
        cls.app.dependency_overrides[auth.get_db] = dependency
        cls.app.dependency_overrides[cls.routes.get_db] = dependency
        cls.client = TestClient(cls.app)

    def setUp(self):
        db.Base.metadata.drop_all(self.engine)
        db.Base.metadata.create_all(self.engine)

    def register(self, email='one@example.com'):
        response = self.client.post('/auth/register', json={
            'email': email, 'password': 'password-123', 'full_name': 'Test Person'})
        self.assertEqual(response.status_code, 201, response.text)
        return {'Authorization': f"Bearer {response.json()['token']}"}

    def test_registration_hashing_normalization_duplicate_and_login(self):
        headers = self.register('ONE@EXAMPLE.COM')
        self.assertEqual(self.client.get('/auth/me', headers=headers).json()['email'], 'one@example.com')
        duplicate = self.client.post('/auth/register', json={
            'email': 'one@example.com', 'password': 'password-123', 'full_name': 'Other'})
        self.assertEqual(duplicate.status_code, 409)
        with self.sessions() as session:
            user = session.query(db.User).one()
            self.assertNotIn('password-123', user.password_hash)
            self.assertEqual(len(session.query(db.AuthSession).one().token_hash), 64)
        for password, status in [('wrong-password', 401), ('password-123', 200)]:
            response = self.client.post('/auth/login', json={'email': 'one@example.com', 'password': password})
            self.assertEqual(response.status_code, status)
        for data in [{'email': 'bad-email', 'password': 'password-123', 'full_name': 'Name'},
                     {'email': 'ok@example.com', 'password': 'short', 'full_name': 'Name'},
                     {'email': 'ok@example.com', 'password': 'password-123', 'full_name': '   '}]:
            self.assertEqual(self.client.post('/auth/register', json=data).status_code, 422)

    def test_auth_required_expiry_and_logout(self):
        for method, path, data in [('GET', '/stats', None), ('GET', '/applicant/42101-1234567-1', None),
                                   ('DELETE', '/applicant/42101-1234567-1', None),
                                   ('POST', '/predict/new', PAYLOAD),
                                   ('POST', '/predict/existing', {**PAYLOAD, 'loan_int_rate': 12})]:
            self.assertEqual(self.client.request(method, path, json=data).status_code, 401)
        headers = self.register()
        self.assertEqual(self.client.post('/auth/logout', headers=headers).status_code, 200)
        self.assertEqual(self.client.get('/auth/me', headers=headers).status_code, 401)
        token = self.client.post('/auth/login', json={'email': 'one@example.com', 'password': 'password-123'}).json()['token']
        with self.sessions() as session:
            session.query(db.AuthSession).update({'expires_at': datetime.utcnow() - timedelta(seconds=1)})
            session.commit()
        self.assertEqual(self.client.get('/auth/me', headers={'Authorization': f'Bearer {token}'}).status_code, 401)

    def test_same_cnic_isolated_profiles_stats_and_deletion(self):
        first, second = self.register(), self.register('two@example.com')
        response = self.client.post('/predict/new', headers=first, json=PAYLOAD)
        self.assertEqual(response.status_code, 200, response.text)
        path = '/applicant/' + PAYLOAD['cnic']
        self.assertEqual(self.client.get(path, headers=second).status_code, 404)
        self.assertEqual(self.client.delete(path, headers=second).status_code, 404)
        response = self.client.post('/predict/existing', headers=second,
                                    json={**PAYLOAD, 'loan_amnt': 20000, 'loan_int_rate': 12})
        self.assertEqual(response.status_code, 200, response.text)
        for headers, amount, rate in [(first, 10000, 11.2), (second, 20000, 12)]:
            profile = self.client.get(path, headers=headers).json()
            self.assertEqual(profile['total_visits'], 1)
            self.assertEqual(profile['last_loan_amnt'], amount)
            self.assertEqual(profile['last_interest_rate'], rate)
            self.assertEqual(self.client.get('/stats', headers=headers).json()['total'], 1)
        self.assertEqual(self.client.delete(path, headers=first).status_code, 200)
        self.assertEqual(self.client.get(path, headers=first).status_code, 404)
        self.assertEqual(self.client.get(path, headers=second).status_code, 200)
        self.assertEqual(self.client.delete(path, headers=second).status_code, 200)
        with self.sessions() as session:
            for model in [db.Applicant, db.Prediction, db.PredictionOwner, db.Input, db.Explanation]:
                self.assertEqual(session.query(model).count(), 0)

    def test_legacy_records_preserved_and_not_exposed(self):
        with self.sessions() as session:
            data = prediction(PAYLOAD)
            db.save_full_prediction(session, PAYLOAD['cnic'], data['prediction_data'], PAYLOAD, [])
        headers = self.register()
        self.assertEqual(self.client.get('/applicant/' + PAYLOAD['cnic'], headers=headers).status_code, 404)
        self.client.post('/predict/new', headers=headers, json=PAYLOAD)
        self.client.delete('/applicant/' + PAYLOAD['cnic'], headers=headers)
        with self.sessions() as session:
            self.assertEqual(session.query(db.Prediction).count(), 1)
            self.assertEqual(session.query(db.Applicant).one().total_visits, 1)

    def test_cnic_normalized_and_save_is_atomic(self):
        headers = self.register()
        response = self.client.post('/predict/new', headers=headers,
                                    json={**PAYLOAD, 'cnic': '4210112345671'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get('/applicant/' + PAYLOAD['cnic'], headers=headers).status_code, 200)
        with self.sessions() as session:
            data = prediction(PAYLOAD)
            with patch.object(db, 'save_explanation', side_effect=RuntimeError('test rollback')):
                with self.assertRaises(RuntimeError):
                    db.save_full_prediction(session, '99999-0000000-0', data['prediction_data'], PAYLOAD, [], user_id=1)
            self.assertEqual(session.query(db.Applicant).count(), 1)
            self.assertEqual(session.query(db.Prediction).count(), 1)


if __name__ == '__main__':
    unittest.main()
