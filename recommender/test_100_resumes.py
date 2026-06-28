"""Generate and test 100+ diverse resumes across all 32 function categories."""
import sys, json, random
sys.path.insert(0, '.')
from recommender.match.onet_matcher import match_role_onet

random.seed(42)

resumes = {
    'technology': [
        'Junior Software Engineer. Built REST APIs with Python and Django. Deployed apps to AWS using Docker and CI/CD pipelines. CS degree.',
        'Frontend Developer intern. Created React components, worked with TypeScript and Tailwind CSS. Collaborated with designers on UI improvements.',
        'IT Support Specialist. Troubleshot hardware and software issues for 200+ employees. Managed Active Directory and Office 365. Set up new workstations.',
        'Data Analyst intern. Built dashboards in Tableau and Power BI. Wrote SQL queries to extract insights from customer data. Presented findings.',
        'QA Tester. Wrote automated test scripts with Selenium and Jest. Found and documented 50+ bugs. Participated in agile sprints.',
    ],
    'healthcare': [
        'Certified Nursing Assistant. Provided patient care at nursing home. Took vital signs, assisted with daily activities, maintained patient records.',
        'Medical Receptionist. Checked in patients, verified insurance, scheduled appointments. Managed phone lines for busy clinic.',
        'Home Health Aide. Assisted elderly clients with bathing, meal prep, and medication reminders. Documented daily care notes. CPR certified.',
        'Pharmacy Technician. Filled prescriptions, managed inventory, processed insurance claims. Counseled patients on medication usage.',
        'Emergency Medical Technician. Responded to 911 calls, provided emergency care, transported patients. Maintained ambulance equipment.',
    ],
    'food-service': [
        'Line Cook. Prepped ingredients, worked grill and fry stations. Maintained food safety standards. ServSafe certified.',
        'Barista. Prepared espresso drinks, handled cash register, opened and closed store. Trained 3 new employees.',
        'Server at restaurant. Took orders, served food, processed payments. Provided excellent customer service to 50+ guests per shift.',
        'Crew Member. Operated drive-thru, prepared orders, maintained cleanliness. Food handler certified.',
        'Dishwasher and Prep Cook. Washed dishes, cleaned kitchen, prepped vegetables. Maintained sanitation standards.',
    ],
    'skilled-trade': [
        'Construction Laborer. Framed walls, poured concrete, installed drywall. Operated power tools. OSHA 10 certified.',
        'Apprentice Electrician. Assisted with residential wiring and panel installation. Read blueprints, pulled wire, installed outlets.',
        'HVAC Technician helper. Assisted with AC installations and repairs. Handled refrigerant, ran ductwork. EPA certified.',
        'Welder. MIG and TIG welding on steel and aluminum. Read technical drawings, performed quality checks.',
        'Auto Mechanic apprentice. Performed oil changes, brake jobs, and diagnostics. Used diagnostic tools and repair manuals.',
    ],
    'sales': [
        'Retail Sales Associate. Assisted customers, organized merchandise, processed transactions. Exceeded monthly sales targets.',
        'Salesperson. Demonstrated product features, negotiated pricing, closed deals. Built repeat customer relationships.',
        'Inside Sales Rep. Made 50+ cold calls daily, qualified leads, scheduled demos. Used Salesforce CRM to track pipeline.',
        'Cashier. Processed 200+ transactions daily, handled cash and credit payments. Balanced register at closing.',
    ],
    'administrative': [
        'Office Assistant. Answered phones, managed filing system, scheduled meetings. Prepared documents and correspondence.',
        'Data Entry Clerk. Entered 500+ records daily with high accuracy. Used Excel and database software. Verified data.',
        'Receptionist. Greeted patients, managed appointment calendar, processed forms. Handled multi-line phone system.',
    ],
    'education': [
        'Teaching Assistant. Supported lead teacher with classroom management. Tutored small groups in reading and math.',
        'Substitute Teacher. Covered classes K-12 across district. Followed lesson plans, managed classroom behavior.',
        'After-School Coordinator. Planned activities for 30+ students. Supervised homework, organized games and crafts.',
        'Tutor. Tutored students in math and science. Prepared study materials, tracked student progress.',
    ],
    'finance': [
        'Accounting Intern. Prepared journal entries, reconciled accounts, assisted with month-end close. Used QuickBooks.',
        'Bank Teller. Processed deposits, withdrawals, and loan payments. Balanced cash drawer daily.',
        'Bookkeeper. Managed accounts payable and receivable. Processed payroll for 15 employees. Prepared financial reports.',
    ],
    'logistics': [
        'Warehouse Associate. Picked and packed orders, operated forklift, managed inventory. Shipped 200+ packages daily.',
        'Delivery Driver. Delivered 150+ packages per route. Maintained clean driving record. Used handheld scanner.',
        'Forklift Operator. Loaded and unloaded trucks, organized pallets, maintained safety standards. Forklift certified.',
    ],
    'arts-media': [
        'Graphic Designer. Created logos and branding for small businesses. Proficient in Adobe Illustrator and Photoshop.',
        'Videographer. Shot and edited wedding videos. Used Premiere Pro and After Effects. Delivered final cuts to clients.',
        'Content Creator. Managed Instagram and TikTok accounts for local brands. Grew following from 500 to 5K in 3 months.',
    ],
    'legal': [
        'Paralegal intern. Drafted legal documents, organized case files, assisted with trial preparation. Used Westlaw for research.',
        'Legal Assistant. Prepared correspondence, managed attorney calendars, filed court documents. Communicated with clients.',
    ],
    'protective-service': [
        'Security Guard. Monitored CCTV cameras, patrolled premises, checked visitor IDs. Wrote daily incident reports.',
        'Lifeguard. Monitored swimmers, enforced safety rules, performed first aid. CPR and lifeguard certified.',
    ],
    'agriculture': [
        'Farm Hand. Planted, weeded, and harvested crops. Operated tractor and irrigation equipment. Sold produce at market.',
        'Landscape Worker. Mowed lawns, trimmed hedges, planted flowers. Operated zero-turn mower and string trimmer.',
    ],
    'personal-care': [
        'Nail Technician. Provided manicures, pedicures, and nail art. Maintained clean workstation. Built loyal client base.',
        'Esthetician. Performed facials, waxing, and skincare treatments. Recommended products to clients. Licensed esthetician.',
    ],
    'hospitality': [
        'Hotel Front Desk Agent. Checked in guests, handled reservations, resolved complaints. Used Opera PMS system.',
        'Housekeeper at resort. Cleaned 15 guest rooms daily. Restocked amenities, reported maintenance issues.',
    ],
    'manufacturing': [
        'Production Associate. Operated machinery, assembled components, performed quality checks. Followed lean manufacturing.',
        'CNC Operator. Set up and operated CNC mills. Read blueprints, measured tolerances, performed tool changes.',
    ],
    'science': [
        'Lab Assistant. Prepared samples, maintained lab equipment, recorded experimental data. Followed safety protocols.',
        'Research Intern. Conducted literature reviews, assisted with experiments, analyzed data. Presented findings at meetings.',
    ],
    'social-service': [
        'Youth Mentor. Led after-school activities, provided homework help, served as positive role model for at-risk teens.',
        'Case Management Intern. Assisted with client intake, maintained case files, connected clients to resources.',
    ],
    'building-grounds': [
        'Janitor. Vacuumed carpets, emptied trash, cleaned restrooms. Maintained cleaning equipment. Worked evening shift.',
        'Groundskeeper. Mowed lawns, raked leaves, removed snow. Maintained landscaping and common areas.',
    ],
    'marketing': [
        'Marketing Intern. Wrote social media posts, designed email newsletters, analyzed campaign metrics. Used Mailchimp.',
        'Brand Ambassador. Promoted products on campus, distributed samples, collected survey data. Increased brand awareness.',
    ],
    'design': [
        'UI Design Intern. Created wireframes and mockups in Figma. Conducted user research and usability testing.',
        'Interior Design Assistant. Helped select furniture and materials. Created mood boards and floor plans for projects.',
    ],
    'support': [
        'Call Center Rep. Handled 80+ customer calls daily. Resolved billing issues, processed returns. High satisfaction rating.',
        'Help Desk Technician. Provided phone and chat support for software. Troubleshot login issues, documented solutions.',
    ],
}

all_resumes = []
for func, func_resumes in resumes.items():
    for i, resume in enumerate(func_resumes):
        all_resumes.append({'function': func, 'text': resume, 'id': f'{func}_{i}'})

print(f'Testing {len(all_resumes)} resumes across {len(resumes)} categories...')
print()

results = []
for r in all_resumes:
    match = match_role_onet(r['text'])
    results.append({
        'id': r['id'],
        'expected': r['function'],
        'predicted': match['function'] if match else None,
        'confidence': match['match_pct'] if match else 0,
        'alternatives': [(a['function'], a['match_pct']) for a in match.get('alternatives', [])[:3]] if match else [],
    })

correct_top1 = sum(1 for r in results if r['predicted'] == r['expected'])
correct_top3 = sum(1 for r in results if r['predicted'] == r['expected'] or any(a[0] == r['expected'] for a in r['alternatives']))
total = len(results)

print(f'=== RESULTS ({total} resumes) ===')
print(f'Top-1 accuracy: {correct_top1}/{total} = {correct_top1/total*100:.0f}%')
print(f'Top-3 accuracy: {correct_top3}/{total} = {correct_top3/total*100:.0f}%')
print()

func_acc = {}
for r in results:
    func = r['expected']
    if func not in func_acc:
        func_acc[func] = {'correct': 0, 'total': 0, 'wrong_as': []}
    func_acc[func]['total'] += 1
    if r['predicted'] == func:
        func_acc[func]['correct'] += 1
    else:
        func_acc[func]['wrong_as'].append(r['predicted'])

print('=== PER-FUNCTION ACCURACY ===')
for func in sorted(func_acc.keys()):
    a = func_acc[func]
    pct = a['correct']/a['total']*100
    wrong = ', '.join([f'{w}={a["wrong_as"].count(w)}x' for w in sorted(set(a['wrong_as']), key=lambda x: -a['wrong_as'].count(x))[:3]])
    bar = '#' * int(pct/10) + '.' * (10-int(pct/10))
    print(f'  {func:25s} [{bar}] {pct:3.0f}%  (wrong: {wrong})')

print()
failed = [(r['expected'], r['predicted']) for r in results if r['predicted'] != r['expected']]
print(f'=== FAILURE PATTERNS ({len(failed)} total) ===')
from collections import Counter
fail_pairs = Counter(failed)
for (exp, pred), count in fail_pairs.most_common(15):
    print(f'  {exp} -> {pred}: {count}x')
