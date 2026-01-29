# GPU Quota Request Guide for Admin

**Project ID:** `piqc-483417`  
**User to Grant Access:** `ammarkhan2264388@gmail.com`

---

## Overview

We need **1 GPU** to test the PIQC tool on GKE. This requires:
1. Requesting GPU quota from Google (you must do this)
2. Granting me permission to proceed with testing

**Estimated Cost:** ~$1-2 for complete testing (using Spot instances)

---

## Step 1: Request GPU Quota (5 minutes)

### 1.1 Open the Quotas Page

Click this link (make sure you're logged in as project owner):

👉 **https://console.cloud.google.com/iam-admin/quotas?project=piqc-483417**

### 1.2 Filter for GPU Quota

1. In the **Filter** box at the top, type: `GPUS_ALL_REGIONS`
2. Press Enter
3. You should see a row showing **"GPUs (all regions)"** with **Limit: 0**

### 1.3 Request Quota Increase

1. **Check the checkbox** next to "GPUs (all regions)"
2. Click the **"EDIT QUOTAS"** button at the top of the page
3. A side panel will open on the right

### 1.4 Fill in the Request Form

| Field | Value |
|-------|-------|
| **New limit** | `1` |
| **Request description** | `Testing PIQC (AI/ML monitoring tool) on GKE with T4 GPU. Short-term testing only, will delete resources after testing.` |

### 1.5 Submit the Request

1. Click **"NEXT"**
2. Verify your contact information
3. Click **"SUBMIT REQUEST"**

### 1.6 Confirmation

- You'll see a confirmation message
- Google will email you when the quota is approved
- **Typical wait time: 24-48 hours** (sometimes faster)

---

## Step 2: Grant Me Permissions (1 minute)

While waiting for quota approval, please grant me the necessary permissions.

### Option A: Using Cloud Console (Easiest)

1. Go to: **https://console.cloud.google.com/iam-admin/iam?project=piqc-483417**
2. Click **"+ GRANT ACCESS"** at the top
3. In **"New principals"**, enter: `ammarkhan2264388@gmail.com`
4. Click **"Select a role"** and choose: `Project > Owner`
5. Click **"Save"**

### Option B: Using Cloud Shell Command

Open Cloud Shell and run:

```bash
gcloud projects add-iam-policy-binding piqc-483417 \
  --member="user:ammarkhan2264388@gmail.com" \
  --role="roles/owner"
```

---

## Step 3: Notify Me

Once you've completed both steps, please let me know:

1. ✅ GPU quota requested (I'll wait for Google's approval email)
2. ✅ Permissions granted (I can start preparing)

---

## What Happens After Quota is Approved?

Once Google approves the GPU quota (you'll get an email), I will:

1. Create a GKE cluster with 1 T4 GPU (~$0.35/hour with Spot pricing)
2. Deploy a test AI model (TinyLlama - small and fast)
3. Run the PIQC tool and generate test outputs
4. **Delete all resources** to stop charges

**Total testing time:** ~1-2 hours  
**Total cost:** ~$1-2

---

## Quick Reference

| Task | Who | Time |
|------|-----|------|
| Request GPU quota | Admin (you) | 5 min |
| Google approves quota | Google | 24-48 hours |
| Grant me permissions | Admin (you) | 1 min |
| Complete testing | Me | 1-2 hours |
| Delete resources | Me | 5 min |

---

## Questions?

If you have any issues with these steps, please let me know and I can help troubleshoot.

Thank you for your help!
