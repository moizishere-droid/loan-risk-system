"""Streamlit landing, persistent accounts, and private loan assessments."""
import html
import os
import re
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st

API_URL = os.getenv('API_URL', 'http://127.0.0.1:8000').rstrip('/')
st.set_page_config(page_title='Loan Assessment', page_icon='🏦', layout='wide')


def navigate(page):
    st.session_state.page = page
    st.rerun()


def clear_account():
    for key in list(st.session_state):
        if key != 'appearance':
            del st.session_state[key]
    st.session_state.page = 'Log in'


def api_request(method, path, **kwargs):
    """Send mutations once, with no automatic prediction/deletion retries."""
    headers = {}
    if st.session_state.get('token'):
        headers['Authorization'] = f"Bearer {st.session_state.token}"
    try:
        response = requests.request(method, f'{API_URL}{path}', headers=headers,
                                    timeout=kwargs.pop('timeout', 60), **kwargs)
        data = response.json()
    except (requests.RequestException, ValueError):
        st.error('The assessment service is unavailable. Please try again in a moment.')
        return None
    if response.status_code == 401 and st.session_state.get('token'):
        clear_account()
        st.session_state.notice = 'Your session has expired. Please log in again.'
        st.rerun()
    if not response.ok:
        detail = data.get('detail', 'We could not complete this request.') if isinstance(data, dict) else ''
        if isinstance(detail, list):
            detail = ' '.join(f"{item.get('loc', ['Input'])[-1]}: {item.get('msg', 'Invalid value')}"
                              for item in detail)
        st.error(detail or 'We could not complete this request.')
        return None
    return data


def heading(eyebrow, title, subtitle=''):
    st.markdown(f'<div class="eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def render_header():
    brand, home, workspace, account, theme = st.columns([4, 1, 1.4, 1.4, 0.6])
    with brand:
        st.markdown('<div class="brand"><span>▥</span> Loan Assessment</div>', unsafe_allow_html=True)
    with home:
        if st.button('Home', use_container_width=True):
            navigate('Home')
    with workspace:
        if st.button('Workspace', use_container_width=True):
            navigate('Workspace' if st.session_state.get('token') else 'Log in')
    with account:
        if st.session_state.get('token'):
            if st.button('Log out', use_container_width=True):
                try:
                    requests.post(f'{API_URL}/auth/logout',
                                  headers={'Authorization': f'Bearer {st.session_state.token}'}, timeout=10)
                except requests.RequestException:
                    pass
                clear_account()
                st.rerun()
        elif st.button('Log in', use_container_width=True):
            navigate('Log in')
    with theme:
        st.button('◐', key='theme_toggle',
                  use_container_width=True, on_click=toggle_appearance)
    st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)


def toggle_appearance():
    st.session_state.appearance = 'Dark' if st.session_state.appearance == 'Light' else 'Light'


def landing():
    left, right = st.columns([1.15, 1], gap='large')
    with left:
        st.markdown('<div class="hero"><div class="eyebrow">Clarity in every assessment</div>'
                    '<h1>Better insight.<br>Confident lending.</h1>'
                    '<p>Assess loan risk, estimate interest rates, and understand the factors '
                    'behind every prediction — in one focused workspace.</p></div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            if st.button('Get started →', type='primary', use_container_width=True):
                navigate('Workspace' if st.session_state.get('token') else 'Sign up')
        with b:
            if st.button('Open workspace', use_container_width=True):
                navigate('Workspace' if st.session_state.get('token') else 'Log in')
        st.caption('New applications · Existing loans · Applicant records')
    with right:
        st.markdown('''<div class="overview"><div class="overline">ASSESSMENT OVERVIEW <span>Illustrative data</span></div>
<div class="pill">Assessment complete</div><h2 class="approved">Approved</h2>
<p>Personal loan · $10,000</p><div class="sample-metrics">
<div><strong>8.4%</strong><small>Default probability</small></div>
<div><strong>11.2%</strong><small>Estimated rate</small></div>
<div><strong>A</strong><small>Loan grade</small></div></div>
<div class="sample-reason">No previous default <span>Lowers risk</span></div>
<div class="sample-reason">Established credit history <span>Lowers risk</span></div>
<small>Understand the decision, beyond a single score.</small></div>''', unsafe_allow_html=True)
    st.divider()
    for column, number, title, text in zip(st.columns(3), ['01 / Assess', '02 / Understand', '03 / Manage'],
            ['One workspace. Two workflows.', 'Reasons you can follow.', 'Keep applicant context close.'],
            ['Evaluate a new applicant or review an existing loan with its assigned interest rate.',
             'See default probability, borrower segment, and the factors shaping each prediction.',
             'Look up your previous assessments and manage applicant records in one place.']):
        with column:
            st.markdown(f'<div class="eyebrow">{number}</div>', unsafe_allow_html=True)
            st.subheader(title)
            st.caption(text)


def authentication(signup=False):
    story, form = st.columns([1, 1], gap='large')
    with story:
        st.markdown('''<div class="auth-story"><div class="eyebrow">Loan Assessment</div>
<h2>Every decision<br>starts with clarity.</h2><p>Your assessment workspace brings applicant details,
predictions, and explanations together.</p><div>01 &nbsp; Assess new and existing loans</div>
<div>02 &nbsp; Understand prediction results</div><div>03 &nbsp; Revisit applicant profiles</div></div>''',
                    unsafe_allow_html=True)
    with form:
        with st.container(border=True):
            st.subheader('Create your account' if signup else 'Welcome back')
            st.caption('Start your loan assessment workspace.' if signup else 'Log in to continue to your workspace.')
            with st.form('signup' if signup else 'login', clear_on_submit=False):
                name = st.text_input('Full name', max_chars=120) if signup else ''
                email = st.text_input('Email address', max_chars=254, placeholder='you@example.com')
                password = st.text_input('Password', type='password', max_chars=128,
                                         help='Use at least 8 characters.')
                confirm = st.text_input('Confirm password', type='password', max_chars=128) if signup else password
                submitted = st.form_submit_button('Create account →' if signup else 'Log in →',
                                                  type='primary', use_container_width=True)
            if submitted:
                if signup and not name.strip():
                    st.error('Enter your full name.')
                elif not email.strip() or len(password) < 8:
                    st.error('Enter your email and a password of at least 8 characters.')
                elif password != confirm:
                    st.error('The passwords do not match.')
                else:
                    payload = {'email': email, 'password': password}
                    if signup:
                        payload['full_name'] = name
                    with st.spinner('Creating your account…' if signup else 'Logging in…'):
                        result = api_request('POST', '/auth/register' if signup else '/auth/login', json=payload)
                    if result:
                        clear_account()
                        st.session_state.token = result['token']
                        st.session_state.user = result['user']
                        navigate('Workspace')
            st.caption('Already have an account?' if signup else 'New here?')
            if st.button('Log in' if signup else 'Create account', key='auth_switch', use_container_width=True):
                navigate('Log in' if signup else 'Sign up')


def input_form(existing=False):
    st.markdown('#### 01 / Personal information')
    a, b = st.columns(2)
    with a:
        cnic = st.text_input('CNIC', placeholder='42101-1234567-1', max_chars=15)
        income = st.number_input('Annual income ($)', min_value=1.0, value=50000.0, step=1000.0)
        home = st.selectbox('Home ownership', ['Rent', 'Own', 'Mortgage', 'Other']).upper()
    with b:
        age = st.number_input('Age', min_value=18, max_value=99, value=30)
        employment = st.number_input('Employment length (years)', min_value=0.0, max_value=80.0, value=5.0, step=0.5)
    st.markdown('#### 02 / Loan details')
    a, b = st.columns(2)
    with a:
        amount = st.number_input('Loan amount ($)', min_value=1.0, value=10000.0, step=500.0)
        intent = st.selectbox('Loan purpose', ['Personal', 'Education', 'Medical', 'Debt consolidation',
                                              'Home improvement', 'Venture']).replace(' ', '').upper()
    with b:
        grade = st.selectbox('Loan grade', list('ABCDEFG'))
        ratio = st.number_input('Loan / annual income', min_value=0.01, max_value=0.99, value=0.20, step=0.01,
                                help='Enter a fraction: 0.20 means 20% of annual income.')
        rate = st.number_input('Assigned interest rate (%)', min_value=0.1, max_value=99.9,
                               value=12.0, step=0.1) if existing else None
    st.markdown('#### 03 / Credit history')
    a, b = st.columns(2)
    with a:
        default = 'Y' if st.selectbox('Previous default', ['No', 'Yes']) == 'Yes' else 'N'
    with b:
        history = st.number_input('Credit history (years)', min_value=0.0, max_value=80.0, value=5.0, step=0.5)
    payload = dict(cnic=cnic.strip(), person_age=age, person_income=income,
                   person_emp_length=employment, person_home_ownership=home, loan_amnt=amount,
                   loan_grade=grade, loan_intent=intent, loan_percent_income=ratio,
                   cb_person_default_on_file=default, cb_person_cred_hist_length=history)
    if existing:
        payload['loan_int_rate'] = rate
    return payload


def valid_cnic(value):
    return bool(re.fullmatch(r'(?:\d{13}|\d{5}-\d{7}-\d)', value))


def canonical_cnic(value):
    digits = value.replace('-', '')
    return f'{digits[:5]}-{digits[5:12]}-{digits[12]}'


def show_result(data, assigned_rate=None):
    cls, reg, cluster = data['classification'], data.get('regression'), data['clustering']
    st.subheader('Assessment results')
    st.caption(f"Prediction #{data['prediction_id']} · {data['workflow'].replace('_', ' ').title()}")
    message = f"{cls['decision'].title()} · Default probability: {cls['default_probability']:.1%}"
    (st.success if cls['decision'] == 'APPROVED' else st.error)(message)
    a, b, c = st.columns(3)
    a.metric('Default probability', f"{cls['default_probability']:.1%}")
    b.metric('Decision threshold', f"{cls['threshold']:.0%}")
    rate = reg['predicted_interest_rate'] if reg else assigned_rate
    c.metric('Predicted rate' if reg else 'Assigned rate', f'{rate:.2f}%' if rate is not None else '—')
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.subheader('Why this decision?')
            reasons = data.get('plain_reasons') or {}
            for reason in reasons.get('decision_reasons', []):
                st.write(f'• {reason}')
            if not reasons.get('decision_reasons'):
                st.caption('Expand the model details below to see contributing factors.')
            if reg and reasons.get('rate_reasons'):
                st.markdown('**Interest rate factors**')
                for reason in reasons['rate_reasons']:
                    st.write(f'• {reason}')
    with b:
        with st.container(border=True):
            st.subheader('Borrower segment')
            st.write(cluster['cluster_label'])
            for label, probability in cluster['probabilities'].items():
                st.caption(f'{label} · {probability:.1%}')
                st.progress(min(1.0, max(0.0, probability)))
    with st.expander('Model explanation details (SHAP)'):
        st.markdown('**Decision factors**')
        st.dataframe([{'Factor': r['readable_name'], 'SHAP impact': r['shap_impact'],
                       'Direction': r['direction']} for r in
                      cls['reasons_for_default'] + cls['reasons_against_default']], use_container_width=True)
        if reg:
            st.markdown('**Interest rate factors**')
            st.dataframe([{'Factor': r['readable_name'], 'SHAP impact': r['shap_impact'],
                           'Direction': r['direction']} for r in
                          reg['reasons_high_rate'] + reg['reasons_low_rate']], use_container_width=True)


def assessment(existing=False):
    heading('Assessment workspace', 'Existing loan' if existing else 'New applicant',
            'Assess risk using an existing loan’s assigned interest rate.' if existing else
            'Build an applicant profile to estimate risk and interest rate.')
    key = 'existing' if existing else 'new'
    with st.container(border=True):
        with st.form(f'{key}_assessment'):
            payload = input_form(existing)
            submitted = st.form_submit_button('Run assessment →', type='primary', use_container_width=True)
        if submitted:
            st.session_state.pop(f'{key}_result', None)
            if not valid_cnic(payload['cnic']):
                st.error('Enter a valid CNIC: 13 digits or 42101-1234567-1.')
            else:
                payload['cnic'] = canonical_cnic(payload['cnic'])
                with st.spinner('Assessing the applicant…'):
                    result = api_request('POST', f'/predict/{key}', json=payload)
                if result:
                    st.session_state[f'{key}_result'] = (result, payload.get('loan_int_rate'))
    if st.session_state.get(f'{key}_result'):
        data, rate = st.session_state[f'{key}_result']
        show_result(data, rate)


def show_profile(data):
    st.subheader('Applicant summary')
    cols = st.columns(4)
    for column, label, value in zip(cols, ['Total visits', 'Approved', 'Rejected', 'Last decision'],
            [data['total_visits'], data['total_approved'], data['total_rejected'], data['last_decision'] or '—']):
        column.metric(label, value)
    st.caption(f"First seen: {data['first_seen']} · Last seen: {data['last_seen']}")
    a, b = st.columns(2)
    a.metric('Last loan amount', f"${data['last_loan_amnt']:,.0f}" if data.get('last_loan_amnt') is not None else '—')
    b.metric('Last interest rate', f"{data['last_interest_rate']:.2f}%" if data.get('last_interest_rate') is not None else '—')
    visit = data.get('last_visit')
    if visit:
        with st.expander('Last visit: inputs and explanations', expanded=True):
            st.write(f"Prediction #{visit['prediction_id']} · {visit['decision']}")
            if visit.get('default_probability') is not None:
                st.caption(f"Default probability: {visit['default_probability']:.1%}")
            if visit.get('inputs'):
                st.dataframe([{'Field': key.replace('_', ' ').title(), 'Value': str(value)}
                              for key, value in visit['inputs'].items()], use_container_width=True)
            if visit.get('explanations'):
                st.dataframe(visit['explanations'], use_container_width=True)


def applicant_records(deleting=False):
    heading('Applicant records', 'Delete applicant' if deleting else 'Applicant profile',
            'Review your applicant records before permanently removing them.' if deleting else
            'Find your previous assessments and the most recent applicant details.')
    key = 'delete' if deleting else 'profile'
    with st.container(border=True):
        with st.form(f'{key}_search'):
            cnic = st.text_input('Applicant CNIC', placeholder='42101-1234567-1', max_chars=15)
            search = st.form_submit_button('Review records' if deleting else 'Find applicant', type='primary')
        if search:
            st.session_state.pop(f'{key}_data', None)
            st.session_state.pop('delete_confirmation', None)
            if not valid_cnic(cnic.strip()):
                st.error('Enter a valid CNIC: 13 digits or 42101-1234567-1.')
            else:
                data = api_request('GET', f'/applicant/{quote(canonical_cnic(cnic.strip()), safe="")}')
                if data:
                    st.session_state[f'{key}_data'] = data
        data = st.session_state.get(f'{key}_data')
        if not data:
            st.caption('Only assessments saved under your account are shown.')
            return
        st.markdown(f"**CNIC: {data['cnic']}**")
        if not deleting:
            show_profile(data)
            return
        st.warning(f"This permanently deletes your {data['total_visits']} assessment(s) for this applicant, "
                   'including their inputs and explanations. This action cannot be undone.')
        st.caption('Records belonging to other accounts are preserved.')
        with st.form('confirm_delete'):
            confirmation = st.text_input('Type DELETE to confirm', key='delete_confirmation')
            confirmed = st.form_submit_button('Permanently delete applicant', type='primary')
        if confirmed:
            if confirmation != 'DELETE':
                st.error('Type DELETE exactly to confirm.')
            elif api_request('DELETE', f'/applicant/{quote(data["cnic"], safe="")}'):
                for state_key in ['delete_data', 'profile_data', 'new_result', 'existing_result']:
                    st.session_state.pop(state_key, None)
                st.session_state.notice = 'Your applicant assessments and related records were deleted.'
                st.rerun()


def select_workflow(section):
    st.session_state.workflow = section


def workspace():
    if not st.session_state.get('token'):
        navigate('Log in')
    user = api_request('GET', '/auth/me', timeout=10)
    if user is None:
        return
    with st.sidebar:
        st.markdown('<div class="eyebrow">Your workspace</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-heading">Loan management</div>', unsafe_allow_html=True)
        section = st.session_state.setdefault('workflow', 'New Applicant')
        for label, key in [('New Applicant', 'nav_new'), ('Existing Loan', 'nav_existing'),
                           ('Applicant Profile', 'nav_profile'), ('Delete Applicant', 'nav_delete')]:
            st.button(label, key=key, use_container_width=True,
                      type='primary' if section == label else 'secondary',
                      on_click=select_workflow, args=(label,))
        name = html.escape(user['full_name'])
        email = html.escape(user['email'])
        initials = html.escape(''.join(part[0] for part in user['full_name'].split()[:2]).upper())
        st.markdown(f'<div class="sidebar-account"><span class="sidebar-avatar">{initials}</span>'
                    f'<div><strong>{name}</strong><div class="sidebar-email">{email}</div></div></div>',
                    unsafe_allow_html=True)
        st.caption('Your assessments are private to your account.')
    if section in ['New Applicant', 'Existing Loan']:
        health = api_request('GET', '/health', timeout=10)
        if health is None:
            return
        if health.get('status') != 'healthy' or not health.get('models') or not all(
                value is True for value in health['models'].values()):
            st.info('The prediction service is starting. Please retry in a moment.')
            st.button('Retry connection')
            return
        assessment(section == 'Existing Loan')
    else:
        applicant_records(section == 'Delete Applicant')


def main():
    st.session_state.setdefault('page', 'Home')
    appearance = st.session_state.setdefault('appearance', 'Light')
    css = (Path(__file__).parent / 'ui' / 'styles.css').read_text(encoding='utf-8')
    palette = '''--lr-bg:#151a18;--lr-paper:#1c2420;--lr-text:#edf4ef;--lr-muted:#adbbb2;
    --lr-line:#36483d;--lr-accent:#99d9c8;--lr-soft:#253e33;--lr-on:#123428;color-scheme:dark;''' if appearance == 'Dark' else '''
    --lr-bg:#f7f9f8;--lr-paper:#fff;--lr-text:#202e28;--lr-muted:#65756c;
    --lr-line:#dfe7e2;--lr-accent:#246f60;--lr-soft:#edf5f1;--lr-on:#fff;color-scheme:light;'''
    st.markdown(f'<style>:root{{{palette}}}{css}</style>', unsafe_allow_html=True)
    render_header()
    if st.session_state.get('notice'):
        st.info(st.session_state.pop('notice'))
    page = st.session_state.page
    if page == 'Home':
        landing()
    elif page in ['Log in', 'Sign up']:
        authentication(page == 'Sign up')
    else:
        workspace()
    st.markdown('<div class="footer">Loan Risk Assessment System <span>Assess · Understand · Manage</span></div>',
                unsafe_allow_html=True)


main()
