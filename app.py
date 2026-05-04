# --- ACTION BAR (DOWNLOAD, SIMPAN DB, HAPUS) ---
        col_csv, col_excel, col_save, col_del = st.columns([1.2, 1.2, 2.5, 1.5])
        
        csv_data = res_df.to_csv(index=False).encode('utf-8')
        col_csv.download_button("⬇️ CSV", csv_data, "hasil_sentimen.csv", "text/csv", use_container_width=True)
        
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
            res_df.to_excel(writer, index=False, sheet_name='Sentimen')
        col_excel.download_button("⬇️ Excel", output_excel.getvalue(), "hasil_sentimen.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        
        # ---> FITUR BARU: SIMPAN KE FIREBASE <---
        if col_save.button("💾 Simpan ke Database"):
            st.info("Sedang menyimpan data ke Firebase, mohon jangan tutup halaman ini...")
            prog_save = st.progress(0)
            total_data = len(res_df)
            
            # Looping untuk menyimpan setiap baris ke Firebase
            for i, row in res_df.iterrows():
                save_to_firebase(row["Text Asli"], row["Sentimen"], row["Keyakinan (%)"])
                # Update progress bar
                prog_save.progress((i + 1) / total_data)
            
            st.success(f"Mantap! {total_data} data berhasil disimpan permanen ke Firebase!")
            time.sleep(2) # Jeda 2 detik biar notifnya terbaca
            
            # Reset tampilan setelah berhasil disimpan agar datanya masuk ke grafik utama
            st.session_state.dataset = None
            st.session_state.page = 0
            st.session_state.page_dashboard = 0
            st.rerun()

        if col_del.button("🗑️ Hapus Hasil"):
            st.session_state.dataset = None
            st.session_state.page = 0
            st.session_state.page_dashboard = 0
            st.rerun()
